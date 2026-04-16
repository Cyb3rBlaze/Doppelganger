import { NextResponse } from "next/server";

import { quoteIdentifier, withPostgres } from "@/lib/postgres";

type GraphRequest = {
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
    const body = (await request.json()) as GraphRequest;
    const connectionString = body.connectionString?.trim();
    const schema = body.schema?.trim();
    const table = body.table?.trim();

    if (!connectionString || !schema || !table) {
      return NextResponse.json(
        { error: "Connection URL, schema, and table are required." },
        { status: 400 },
      );
    }

    if (schema !== "public" || !["document_chunks", "memory_nodes"].includes(table)) {
      return NextResponse.json(
        { error: "Graph mode is only available for public.document_chunks and public.memory_nodes." },
        { status: 400 },
      );
    }

    const payload = await withPostgres(connectionString, async (client) => {
      const safeTableReference = `${quoteIdentifier(schema)}.${quoteIdentifier(table)}`;

      if (table === "document_chunks") {
        const nodesResult = await client.query<Record<string, unknown>>(
          `
            select *
            from ${safeTableReference}
            order by document_id asc, chunk_index asc
            limit 200
          `,
        );

        return {
          nodes: nodesResult.rows.map((row) =>
            Object.fromEntries(
              Object.entries(row).map(([key, value]) => [key, normalizeValue(value)]),
            ),
          ),
          edges: [],
        };
      }

      const nodesResult = await client.query<Record<string, unknown>>(
        `
          select *
          from ${safeTableReference}
          order by created_at desc, node_id asc
          limit 200
        `,
      );

      const nodeIds = nodesResult.rows
        .map((row) => row.node_id)
        .filter((value): value is string => typeof value === "string");

      if (nodeIds.length === 0) {
        return { nodes: [], edges: [] };
      }

      const edgesResult = await client.query<Record<string, unknown>>(
        `
          select *
          from public.memory_edges
          where source_node_id = any($1::text[])
            and target_node_id = any($1::text[])
          order by created_at asc, id asc
        `,
        [nodeIds],
      );

      return {
        nodes: nodesResult.rows.map((row) =>
          Object.fromEntries(
            Object.entries(row).map(([key, value]) => [key, normalizeValue(value)]),
          ),
        ),
        edges: edgesResult.rows.map((row) =>
          Object.fromEntries(
            Object.entries(row).map(([key, value]) => [key, normalizeValue(value)]),
          ),
        ),
      };
    });

    return NextResponse.json(payload);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Could not build graph data for this table.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
