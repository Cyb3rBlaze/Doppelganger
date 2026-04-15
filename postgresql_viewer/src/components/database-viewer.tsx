"use client";

import { useState } from "react";

import styles from "./database-viewer.module.css";

type TableEntry = {
  schema: string;
  name: string;
  kind: string;
  columnCount: number;
};

type ColumnEntry = {
  name: string;
  dataType: string;
};

type DatabasePayload = {
  database: string;
  version: string;
  tables: TableEntry[];
};

type PreviewPayload = {
  columns: ColumnEntry[];
  rows: Record<string, unknown>[];
};

function tableKey(table: Pick<TableEntry, "schema" | "name">) {
  return `${table.schema}.${table.name}`;
}

function formatValue(value: unknown) {
  if (value === null) {
    return "null";
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return JSON.stringify(value);
}

export function DatabaseViewer() {
  const [connectionInput, setConnectionInput] = useState("");
  const [activeConnectionString, setActiveConnectionString] = useState("");
  const [database, setDatabase] = useState<DatabasePayload | null>(null);
  const [selectedTable, setSelectedTable] = useState<TableEntry | null>(null);
  const [preview, setPreview] = useState<PreviewPayload | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function loadPreview(table: TableEntry, connectionString: string) {
    setSelectedTable(table);
    setIsLoadingPreview(true);
    setErrorMessage("");

    try {
      const response = await fetch("/api/preview", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          connectionString,
          schema: table.schema,
          table: table.name,
        }),
      });

      const payload = (await response.json()) as PreviewPayload & { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "Could not load the table preview.");
      }

      setPreview(payload);
    } catch (error) {
      setPreview(null);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not load the table preview.",
      );
    } finally {
      setIsLoadingPreview(false);
    }
  }

  async function handleConnect(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const connectionString = connectionInput.trim();

    if (!connectionString) {
      setErrorMessage("Paste a PostgreSQL connection URL to continue.");
      return;
    }

    setIsConnecting(true);
    setErrorMessage("");
    setDatabase(null);
    setPreview(null);
    setSelectedTable(null);

    try {
      const response = await fetch("/api/connect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ connectionString }),
      });

      const payload = (await response.json()) as DatabasePayload & { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "Could not connect to the database.");
      }

      setActiveConnectionString(connectionString);
      setDatabase(payload);

      if (payload.tables.length > 0) {
        await loadPreview(payload.tables[0], connectionString);
      }
    } catch (error) {
      setActiveConnectionString("");
      setErrorMessage(
        error instanceof Error ? error.message : "Could not connect to the database.",
      );
    } finally {
      setIsConnecting(false);
    }
  }

  return (
    <main className={styles.shell}>
      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <p className={styles.kicker}>Postgres Viewer</p>
          <h1>Paste a database URL and browse your tables in seconds.</h1>
          <p className={styles.subcopy}>
            The connection string stays out of the URL. Each action sends it directly
            to the server to inspect schemas and preview up to 100 rows.
          </p>
        </div>
        <form className={styles.connectCard} onSubmit={handleConnect}>
          <label className={styles.fieldLabel} htmlFor="connectionString">
            PostgreSQL URL
          </label>
          <textarea
            id="connectionString"
            className={styles.textarea}
            rows={4}
            placeholder="postgresql://user:password@host:5432/database?sslmode=require"
            value={connectionInput}
            onChange={(event) => setConnectionInput(event.target.value)}
          />
          <button className={styles.connectButton} disabled={isConnecting}>
            {isConnecting ? "Connecting..." : "Connect"}
          </button>
          <p className={styles.helperText}>
            Works with hosted URLs too. `sslmode=require` is supported.
          </p>
        </form>
      </section>

      {errorMessage ? <p className={styles.errorBanner}>{errorMessage}</p> : null}

      <section className={styles.viewer}>
        <aside className={styles.sidebar}>
          <div className={styles.sidebarHeader}>
            <div>
              <p className={styles.sidebarEyebrow}>Database</p>
              <h2>{database?.database ?? "Not connected"}</h2>
            </div>
            {database ? (
              <span className={styles.tableCount}>{database.tables.length} objects</span>
            ) : null}
          </div>

          <div className={styles.tableList}>
            {database?.tables.length ? (
              database.tables.map((table) => {
                const isActive =
                  selectedTable !== null && tableKey(table) === tableKey(selectedTable);

                return (
                  <button
                    key={tableKey(table)}
                    className={isActive ? styles.tableButtonActive : styles.tableButton}
                    onClick={() => loadPreview(table, activeConnectionString)}
                    type="button"
                  >
                    <span className={styles.tableName}>
                      {table.schema}.{table.name}
                    </span>
                    <span className={styles.tableMeta}>
                      {table.kind} · {table.columnCount} columns
                    </span>
                  </button>
                );
              })
            ) : (
              <p className={styles.emptyState}>
                Connect to a database to see tables and views here.
              </p>
            )}
          </div>
        </aside>

        <section className={styles.previewPanel}>
          <div className={styles.previewHeader}>
            <div>
              <p className={styles.sidebarEyebrow}>Preview</p>
              <h2>
                {selectedTable
                  ? `${selectedTable.schema}.${selectedTable.name}`
                  : "Select a table"}
              </h2>
            </div>
            {selectedTable ? (
              <span className={styles.limitPill}>Showing up to 100 rows</span>
            ) : null}
          </div>

          {database ? (
            <p className={styles.versionText}>{database.version}</p>
          ) : null}

          {isLoadingPreview ? (
            <p className={styles.emptyState}>Loading preview...</p>
          ) : preview && selectedTable ? (
            preview.rows.length > 0 ? (
              <div className={styles.tableWrap}>
                <table className={styles.dataTable}>
                  <thead>
                    <tr>
                      {preview.columns.map((column) => (
                        <th key={column.name}>
                          <span>{column.name}</span>
                          <small>{column.dataType}</small>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, rowIndex) => (
                      <tr key={`${tableKey(selectedTable)}-${rowIndex}`}>
                        {preview.columns.map((column) => (
                          <td key={`${rowIndex}-${column.name}`}>
                            <code>{formatValue(row[column.name])}</code>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className={styles.emptyState}>
                This object is queryable, but it returned no rows in the first 100.
              </p>
            )
          ) : (
            <p className={styles.emptyState}>
              Pick a table or view to preview its rows.
            </p>
          )}
        </section>
      </section>
    </main>
  );
}
