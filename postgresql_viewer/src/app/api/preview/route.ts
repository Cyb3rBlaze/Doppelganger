import { NextResponse } from "next/server";

import { quoteIdentifier, withPostgres } from "@/lib/postgres";

type PreviewRequest = {
  connectionString?: string;
  schema?: string;
  table?: string;
};

function normalizeValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return null;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (Buffer.isBuffer(value)) {
    return `Buffer(${value.length} bytes)`;
  }

  if (Array.isArray(value)) {
    return value.map(normalizeValue);
  }

  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [
        key,
        normalizeValue(nestedValue),
      ]),
    );
  }

  return value;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as PreviewRequest;
    const connectionString = body.connectionString?.trim();
    const schema = body.schema?.trim();
    const table = body.table?.trim();

    if (!connectionString || !schema || !table) {
      return NextResponse.json(
        { error: "Connection URL, schema, and table are required." },
        { status: 400 },
      );
    }

    const payload = await withPostgres(connectionString, async (client) => {
      const columns = (
        await client.query<{
          name: string;
          dataType: string;
        }>(
          `
            select
              column_name as name,
              data_type as "dataType"
            from information_schema.columns
            where table_schema = $1
              and table_name = $2
            order by ordinal_position
          `,
          [schema, table],
        )
      ).rows;

      const safeTableReference = `${quoteIdentifier(schema)}.${quoteIdentifier(table)}`;
      const previewResult = await client.query<Record<string, unknown>>(
        `select * from ${safeTableReference} limit 100`,
      );

      return {
        columns,
        rows: previewResult.rows.map((row: Record<string, unknown>) =>
          Object.fromEntries(
            Object.entries(row).map(([key, value]) => [key, normalizeValue(value)]),
          ),
        ),
      };
    });

    return NextResponse.json(payload);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Could not preview this table.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
