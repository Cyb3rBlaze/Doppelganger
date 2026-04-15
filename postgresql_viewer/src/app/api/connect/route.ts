import { NextResponse } from "next/server";

import { withPostgres } from "@/lib/postgres";

type ConnectRequest = {
  connectionString?: string;
};

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as ConnectRequest;
    const connectionString = body.connectionString?.trim();

    if (!connectionString) {
      return NextResponse.json(
        { error: "A PostgreSQL connection URL is required." },
        { status: 400 },
      );
    }

    const payload = await withPostgres(connectionString, async (client) => {
      const [{ database, version }] = (
        await client.query<{
          database: string;
          version: string;
        }>("select current_database() as database, version() as version")
      ).rows;

      const tables = (
        await client.query<{
          schema: string;
          name: string;
          kind: string;
          columnCount: number;
        }>(`
          select
            n.nspname as schema,
            c.relname as name,
            case c.relkind
              when 'r' then 'table'
              when 'p' then 'partitioned table'
              when 'v' then 'view'
              when 'm' then 'materialized view'
              when 'f' then 'foreign table'
              else c.relkind::text
            end as kind,
            (
              select count(*)::int
              from information_schema.columns cols
              where cols.table_schema = n.nspname
                and cols.table_name = c.relname
            ) as "columnCount"
          from pg_class c
          join pg_namespace n on n.oid = c.relnamespace
          where n.nspname not in ('pg_catalog', 'information_schema')
            and c.relkind in ('r', 'p', 'v', 'm', 'f')
          order by n.nspname, c.relname
        `)
      ).rows;

      return {
        database,
        version,
        tables,
      };
    });

    return NextResponse.json(payload);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Could not connect to the database.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
