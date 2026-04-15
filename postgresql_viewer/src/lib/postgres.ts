import { Client, type ClientConfig } from "pg";

function isSslEnabled(connectionUrl: URL) {
  const ssl = connectionUrl.searchParams.get("ssl");
  const sslMode = connectionUrl.searchParams.get("sslmode")?.toLowerCase();

  if (ssl === "true") {
    return true;
  }

  if (!sslMode || sslMode === "disable") {
    return false;
  }

  return true;
}

function shouldRejectUnauthorized(connectionUrl: URL) {
  const sslMode = connectionUrl.searchParams.get("sslmode")?.toLowerCase();

  return sslMode === "verify-ca" || sslMode === "verify-full";
}

function toClientConfig(connectionString: string): ClientConfig {
  const connectionUrl = new URL(connectionString);

  if (!["postgres:", "postgresql:"].includes(connectionUrl.protocol)) {
    throw new Error("Please enter a valid postgres:// or postgresql:// URL.");
  }

  const config: ClientConfig = {
    connectionString,
    connectionTimeoutMillis: 5000,
    query_timeout: 10000,
    statement_timeout: 10000,
  };

  if (isSslEnabled(connectionUrl)) {
    config.ssl = {
      rejectUnauthorized: shouldRejectUnauthorized(connectionUrl),
    };
  }

  return config;
}

export function quoteIdentifier(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

export async function withPostgres<T>(
  connectionString: string,
  callback: (client: Client) => Promise<T>,
) {
  const client = new Client(toClientConfig(connectionString));

  await client.connect();

  try {
    return await callback(client);
  } finally {
    await client.end();
  }
}
