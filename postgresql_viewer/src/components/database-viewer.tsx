"use client";

import { useMemo, useState } from "react";

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

type GraphPayload = {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
};

type GraphNode = {
  id: string;
  label: string;
  title: string;
  x: number;
  y: number;
  radius: number;
  scoreLabel: string;
};

type GraphEdge = {
  id: string;
  sourceId: string;
  targetId: string;
  label: string;
  score: number;
};

type PreviewMode = "rows" | "graph";

const MAX_STRING_PREVIEW = 180;
const MAX_OBJECT_PREVIEW = 220;
const MAX_ARRAY_PREVIEW_ITEMS = 6;
const EXPANDED_PATH_COLUMNS = new Set(["source_path"]);
const GRAPH_WIDTH = 1240;
const GRAPH_HEIGHT = 720;
const GRAPH_PADDING = 48;
const GRAPH_TABLES = new Set(["document_chunks", "memory_nodes"]);

function tableKey(table: Pick<TableEntry, "schema" | "name">) {
  return `${table.schema}.${table.name}`;
}

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength - 1)}…`;
}

function formatArrayValue(value: unknown[]) {
  if (value.length === 0) {
    return "[]";
  }

  const previewItems = value.slice(0, MAX_ARRAY_PREVIEW_ITEMS).map((item) => {
    if (typeof item === "number") {
      return Number(item.toFixed(6)).toString();
    }

    return truncateText(formatValue(item), 32);
  });

  const suffix = value.length > MAX_ARRAY_PREVIEW_ITEMS ? ", …" : "";
  return `Array(${value.length}) [${previewItems.join(", ")}${suffix}]`;
}

function formatObjectValue(value: Record<string, unknown>) {
  return truncateText(JSON.stringify(value), MAX_OBJECT_PREVIEW);
}

function parseEmbedding(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    const numbers = value
      .map((item) => (typeof item === "number" ? item : Number(item)))
      .filter((item) => Number.isFinite(item));
    return numbers.length > 0 ? numbers : null;
  }

  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
    return null;
  }

  const numbers = trimmed
    .slice(1, -1)
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((item) => Number.isFinite(item));

  return numbers.length > 0 ? numbers : null;
}

function parseConnectedNodes(value: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(value)) {
    return value.filter((item) => typeof item === "object" && item !== null) as Array<
      Record<string, unknown>
    >;
  }

  if (typeof value !== "string") {
    return [];
  }

  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter((item) => typeof item === "object" && item !== null)
      : [];
  } catch {
    return [];
  }
}

function parseStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }

  if (typeof value !== "string") {
    return [];
  }

  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function parseSignals(value: unknown): Record<string, number> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .map(([key, nestedValue]) => [key, Number(nestedValue)])
        .filter((entry): entry is [string, number] => Number.isFinite(entry[1])),
    );
  }

  if (typeof value !== "string") {
    return {};
  }

  try {
    const parsed = JSON.parse(value);
    return parseSignals(parsed);
  } catch {
    return {};
  }
}

function scaleCoordinate(
  value: number,
  min: number,
  max: number,
  outputMin: number,
  outputMax: number,
) {
  if (max === min) {
    return (outputMin + outputMax) / 2;
  }

  const ratio = (value - min) / (max - min);
  return outputMin + ratio * (outputMax - outputMin);
}

function dotProduct(left: number[], right: number[]) {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    sum += left[index] * right[index];
  }
  return sum;
}

function vectorMagnitude(values: number[]) {
  return Math.sqrt(dotProduct(values, values));
}

function normalizeVector(values: number[]) {
  const magnitude = vectorMagnitude(values);
  if (magnitude === 0) {
    return values.map(() => 0);
  }

  return values.map((value) => value / magnitude);
}

function subtractScaledOuterProduct(
  matrix: number[][],
  scores: number[],
  loadings: number[],
) {
  for (let rowIndex = 0; rowIndex < matrix.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < loadings.length; columnIndex += 1) {
      matrix[rowIndex][columnIndex] -= scores[rowIndex] * loadings[columnIndex];
    }
  }
}

function buildPcaProjection(embeddings: number[][]) {
  if (embeddings.length === 0) {
    return [];
  }

  const dimensionCount = embeddings[0].length;
  const centered = embeddings.map((embedding) => [...embedding]);
  const means = new Array<number>(dimensionCount).fill(0);

  for (const embedding of centered) {
    for (let index = 0; index < dimensionCount; index += 1) {
      means[index] += embedding[index];
    }
  }

  for (let index = 0; index < dimensionCount; index += 1) {
    means[index] /= centered.length;
  }

  for (const embedding of centered) {
    for (let index = 0; index < dimensionCount; index += 1) {
      embedding[index] -= means[index];
    }
  }

  const scoresByComponent: number[][] = [];
  const maxComponents = Math.min(2, centered.length, dimensionCount);

  for (let componentIndex = 0; componentIndex < maxComponents; componentIndex += 1) {
    let scores = centered.map((embedding) => embedding[componentIndex] ?? 0);
    if (vectorMagnitude(scores) === 0) {
      scores = centered.map((embedding) =>
        embedding.reduce((sum, value) => sum + value, 0),
      );
    }

    let previousScores = new Array<number>(scores.length).fill(Number.POSITIVE_INFINITY);

    for (let iteration = 0; iteration < 25; iteration += 1) {
      const scoreMagnitude = dotProduct(scores, scores);
      if (scoreMagnitude === 0) {
        break;
      }

      const loadings = new Array<number>(dimensionCount).fill(0);
      for (let columnIndex = 0; columnIndex < dimensionCount; columnIndex += 1) {
        let numerator = 0;
        for (let rowIndex = 0; rowIndex < centered.length; rowIndex += 1) {
          numerator += centered[rowIndex][columnIndex] * scores[rowIndex];
        }
        loadings[columnIndex] = numerator / scoreMagnitude;
      }

      const normalizedLoadings = normalizeVector(loadings);
      const nextScores = centered.map((embedding) => dotProduct(embedding, normalizedLoadings));
      const delta = nextScores.reduce(
        (sum, value, index) => sum + Math.abs(value - previousScores[index]),
        0,
      );

      previousScores = scores;
      scores = nextScores;

      if (delta < 1e-6) {
        break;
      }
    }

    const scoreMagnitude = dotProduct(scores, scores);
    if (scoreMagnitude === 0) {
      scoresByComponent.push(new Array<number>(centered.length).fill(0));
      continue;
    }

    const loadings = new Array<number>(dimensionCount).fill(0);
    for (let columnIndex = 0; columnIndex < dimensionCount; columnIndex += 1) {
      let numerator = 0;
      for (let rowIndex = 0; rowIndex < centered.length; rowIndex += 1) {
        numerator += centered[rowIndex][columnIndex] * scores[rowIndex];
      }
      loadings[columnIndex] = numerator / scoreMagnitude;
    }

    const normalizedLoadings = normalizeVector(loadings);
    const finalizedScores = centered.map((embedding) => dotProduct(embedding, normalizedLoadings));
    scoresByComponent.push(finalizedScores);
    subtractScaledOuterProduct(centered, finalizedScores, normalizedLoadings);
  }

  return embeddings.map((_, index) => ({
    x: scoresByComponent[0]?.[index] ?? 0,
    y: scoresByComponent[1]?.[index] ?? 0,
  }));
}

function hashString(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function computeFallbackPosition(id: string, index: number) {
  const hash = hashString(id);
  const angle = ((hash % 360) * Math.PI) / 180;
  const radius = 110 + (index % 12) * 24;
  return {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle) * radius,
  };
}

function buildGraphData(
  tableName: string | undefined,
  rows: Record<string, unknown>[],
  edgeRows: Record<string, unknown>[] = [],
) {
  const rawNodes = rows
    .map((row, index) => {
      if (tableName === "memory_nodes") {
        const nodeId =
          typeof row.node_id === "string" ? row.node_id : `node-${index}`;
        const title =
          typeof row.title === "string" && row.title.trim()
            ? row.title.trim()
            : nodeId;
        const nodeType = typeof row.node_type === "string" ? row.node_type : "memory_node";
        const contentPreview =
          typeof row.content === "string" ? truncateText(row.content, 28) : nodeId;
        return {
          id: nodeId,
          label: title === nodeId ? `${nodeType}: ${contentPreview}` : title,
          title: `${nodeId} (${nodeType})`,
          embedding: parseEmbedding(row.embedding),
          scoreLabel: nodeType,
          radius:
            nodeType === "session_summary"
              ? 11
              : nodeType === "document_chunk"
                ? 8
                : 7,
          connectedNodes: [] as Array<Record<string, unknown>>,
        };
      }

      const embedding = parseEmbedding(row.embedding);
      const chunkId =
        typeof row.chunk_id === "string" ? row.chunk_id : `row-${index}`;
      const windowStart =
        typeof row.window_start_chunk_index === "number"
          ? row.window_start_chunk_index
          : Number(row.window_start_chunk_index ?? 0);
      const windowEnd =
        typeof row.window_end_chunk_index === "number"
          ? row.window_end_chunk_index
          : Number(row.window_end_chunk_index ?? windowStart);
      const span = Number.isFinite(windowEnd - windowStart)
        ? Math.max(1, windowEnd - windowStart + 1)
        : 1;

      return {
        id: chunkId,
        label:
          typeof row.title === "string" && row.title.trim()
            ? row.title.trim()
            : chunkId,
        title: chunkId,
        embedding,
        scoreLabel:
          typeof row.score === "number"
            ? row.score.toFixed(3)
            : typeof row.score === "string"
              ? row.score
              : "",
        radius: Math.min(14, 5 + span * 1.4),
        connectedNodes: parseConnectedNodes(row.connected_nodes),
      };
    })
    .filter((node): node is NonNullable<typeof node> => node !== null);

  if (rawNodes.length === 0) {
    return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] };
  }

  const seededNodes = rawNodes.filter(
    (node): node is (typeof rawNodes)[number] & { embedding: number[] } =>
      Array.isArray(node.embedding) && node.embedding.length > 0,
  );
  const projectedSeedPoints = buildPcaProjection(seededNodes.map((node) => node.embedding));
  const positionedPoints = new Map<string, { x: number; y: number }>();
  for (let index = 0; index < seededNodes.length; index += 1) {
    positionedPoints.set(seededNodes[index].id, projectedSeedPoints[index]);
  }

  const pendingNodes = rawNodes.filter((node) => !positionedPoints.has(node.id));
  const candidateEdges =
    tableName === "memory_nodes"
      ? edgeRows
          .map((edgeRow, index) => {
            const sourceId =
              typeof edgeRow.source_node_id === "string"
                ? edgeRow.source_node_id
                : `source-${index}`;
            const targetId =
              typeof edgeRow.target_node_id === "string"
                ? edgeRow.target_node_id
                : `target-${index}`;
            return {
              sourceId,
              targetId,
              score:
                typeof edgeRow.score === "number"
                  ? edgeRow.score
                  : Number(edgeRow.score ?? 0),
              label: parseStringArray(edgeRow.edge_types).join(", "),
              signals: parseSignals(edgeRow.signals),
            };
          })
          .filter((edge) => edge.sourceId && edge.targetId)
      : [];

  for (let pass = 0; pass < 3; pass += 1) {
    for (let index = 0; index < pendingNodes.length; index += 1) {
      const node = pendingNodes[index];
      if (positionedPoints.has(node.id)) {
        continue;
      }

      const neighborPositions = candidateEdges
        .flatMap((edge) => {
          if (edge.sourceId === node.id) {
            return positionedPoints.has(edge.targetId)
              ? [positionedPoints.get(edge.targetId)!]
              : [];
          }
          if (edge.targetId === node.id) {
            return positionedPoints.has(edge.sourceId)
              ? [positionedPoints.get(edge.sourceId)!]
              : [];
          }
          return [];
        });

      if (neighborPositions.length === 0) {
        continue;
      }

      const averageX =
        neighborPositions.reduce((sum, point) => sum + point.x, 0) / neighborPositions.length;
      const averageY =
        neighborPositions.reduce((sum, point) => sum + point.y, 0) / neighborPositions.length;
      const fallback = computeFallbackPosition(node.id, index);
      positionedPoints.set(node.id, {
        x: averageX + fallback.x * 0.08,
        y: averageY + fallback.y * 0.08,
      });
    }
  }

  pendingNodes.forEach((node, index) => {
    if (!positionedPoints.has(node.id)) {
      positionedPoints.set(node.id, computeFallbackPosition(node.id, index));
    }
  });

  const projectedXs = rawNodes.map((node) => positionedPoints.get(node.id)?.x ?? 0);
  const projectedYs = rawNodes.map((node) => positionedPoints.get(node.id)?.y ?? 0);
  const minX = Math.min(...projectedXs);
  const maxX = Math.max(...projectedXs);
  const minY = Math.min(...projectedYs);
  const maxY = Math.max(...projectedYs);

  const nodes: GraphNode[] = rawNodes.map((node, index) => ({
    id: node.id,
    label: truncateText(node.label, 30),
    title: node.title,
    x: scaleCoordinate(
      positionedPoints.get(node.id)?.x ?? computeFallbackPosition(node.id, index).x,
      minX,
      maxX,
      GRAPH_PADDING,
      GRAPH_WIDTH - GRAPH_PADDING,
    ),
    y: scaleCoordinate(
      positionedPoints.get(node.id)?.y ?? computeFallbackPosition(node.id, index).y,
      minY,
      maxY,
      GRAPH_PADDING,
      GRAPH_HEIGHT - GRAPH_PADDING,
    ),
    radius: node.radius,
    scoreLabel: node.scoreLabel,
  }));

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const seenEdges = new Set<string>();
  const edges: GraphEdge[] = [];

  if (tableName === "memory_nodes") {
    for (const edge of candidateEdges) {
      if (!nodeById.has(edge.sourceId) || !nodeById.has(edge.targetId)) {
        continue;
      }
      const edgeId = [edge.sourceId, edge.targetId].sort().join("::");
      if (seenEdges.has(edgeId)) {
        continue;
      }
      seenEdges.add(edgeId);
      edges.push({
        id: edgeId,
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        label: edge.label,
        score: Number.isFinite(edge.score) ? edge.score : 0,
      });
    }
  } else {
    for (const node of rawNodes) {
      for (const connectedNode of node.connectedNodes) {
        const targetId =
          typeof connectedNode.chunk_id === "string" ? connectedNode.chunk_id : "";
        if (!targetId || !nodeById.has(targetId)) {
          continue;
        }

        const edgeId = [node.id, targetId].sort().join("::");
        if (seenEdges.has(edgeId)) {
          continue;
        }
        seenEdges.add(edgeId);

        const score =
          typeof connectedNode.score === "number"
            ? connectedNode.score
            : Number(connectedNode.score ?? 0);
        const edgeTypes = Array.isArray(connectedNode.edge_types)
          ? connectedNode.edge_types.filter((item): item is string => typeof item === "string")
          : [];

        edges.push({
          id: edgeId,
          sourceId: node.id,
          targetId,
          label: edgeTypes.join(", "),
          score: Number.isFinite(score) ? score : 0,
        });
      }
    }
  }

  return { nodes, edges };
}

function formatValue(value: unknown, columnName?: string) {
  if (value === null) {
    return "null";
  }

  if (typeof value === "string") {
    if (columnName && EXPANDED_PATH_COLUMNS.has(columnName)) {
      return value;
    }
    return truncateText(value, MAX_STRING_PREVIEW);
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    return formatArrayValue(value);
  }

  if (typeof value === "object") {
    return formatObjectValue(value as Record<string, unknown>);
  }

  return truncateText(String(value), MAX_OBJECT_PREVIEW);
}

export function DatabaseViewer() {
  const [connectionInput, setConnectionInput] = useState("");
  const [activeConnectionString, setActiveConnectionString] = useState("");
  const [database, setDatabase] = useState<DatabasePayload | null>(null);
  const [selectedTable, setSelectedTable] = useState<TableEntry | null>(null);
  const [preview, setPreview] = useState<PreviewPayload | null>(null);
  const [graphPayload, setGraphPayload] = useState<GraphPayload | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [previewMode, setPreviewMode] = useState<PreviewMode>("rows");

  const canRenderGraph =
    selectedTable?.schema === "public" &&
    GRAPH_TABLES.has(selectedTable?.name ?? "") &&
    Boolean(preview?.rows.length);

  const graphData = useMemo(
    () =>
      buildGraphData(
        selectedTable?.name,
        graphPayload?.nodes ?? preview?.rows ?? [],
        graphPayload?.edges ?? [],
      ),
    [graphPayload, preview, selectedTable],
  );

  async function loadGraphData(table: TableEntry, connectionString: string) {
    if (!(table.schema === "public" && GRAPH_TABLES.has(table.name))) {
      setGraphPayload(null);
      return;
    }

    const response = await fetch("/api/graph", {
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

    const payload = (await response.json()) as GraphPayload & { error?: string };
    if (!response.ok) {
      throw new Error(payload.error ?? "Could not load the graph preview.");
    }

    setGraphPayload(payload);
  }

  async function loadPreview(table: TableEntry, connectionString: string) {
    setSelectedTable(table);
    setIsLoadingPreview(true);
    setErrorMessage("");
    setPreviewMode("rows");
    setGraphPayload(null);

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
      await loadGraphData(table, connectionString);
    } catch (error) {
      setPreview(null);
      setGraphPayload(null);
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

          {canRenderGraph ? (
            <div className={styles.viewModeTabs}>
              <button
                className={
                  previewMode === "rows" ? styles.viewModeButtonActive : styles.viewModeButton
                }
                onClick={() => setPreviewMode("rows")}
                type="button"
              >
                Rows
              </button>
              <button
                className={
                  previewMode === "graph" ? styles.viewModeButtonActive : styles.viewModeButton
                }
                onClick={() => setPreviewMode("graph")}
                type="button"
              >
                Graph
              </button>
            </div>
          ) : null}

          {database ? (
            <p className={styles.versionText}>{database.version}</p>
          ) : null}

          {isLoadingPreview ? (
            <p className={styles.emptyState}>Loading preview...</p>
          ) : preview && selectedTable ? (
            preview.rows.length > 0 ? (
              previewMode === "graph" && canRenderGraph ? (
                graphData.nodes.length > 0 ? (
                  <div className={styles.graphPanel}>
                    <div className={styles.graphLegend}>
                      <span>{graphData.nodes.length} nodes</span>
                      <span>{graphData.edges.length} edges</span>
                      <span>
                        {selectedTable?.name === "memory_nodes"
                          ? "PCA for embedded nodes, neighbor placement for message graph nodes"
                          : "PCA projection from full embeddings"}
                      </span>
                    </div>
                    <div className={styles.graphWrap}>
                      <svg
                        aria-label="Embedding graph"
                        className={styles.graphSvg}
                        viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
                      >
                        <rect
                          x="0"
                          y="0"
                          width={GRAPH_WIDTH}
                          height={GRAPH_HEIGHT}
                          rx="24"
                          className={styles.graphBackdrop}
                        />
                        {graphData.edges.map((edge) => {
                          const source = graphData.nodes.find((node) => node.id === edge.sourceId);
                          const target = graphData.nodes.find((node) => node.id === edge.targetId);
                          if (!source || !target) {
                            return null;
                          }

                          return (
                            <line
                              key={edge.id}
                              x1={source.x}
                              y1={source.y}
                              x2={target.x}
                              y2={target.y}
                              className={styles.graphEdge}
                              strokeWidth={1 + Math.min(2.5, edge.score * 2)}
                            >
                              <title>
                                {`${source.title} ↔ ${target.title}\n${edge.label} (${edge.score.toFixed(3)})`}
                              </title>
                            </line>
                          );
                        })}
                        {graphData.nodes.map((node) => (
                          <g key={node.id}>
                            <circle
                              cx={node.x}
                              cy={node.y}
                              r={node.radius}
                              className={styles.graphNode}
                            >
                              <title>
                                {`${node.title}\nscore: ${node.scoreLabel || "n/a"}`}
                              </title>
                            </circle>
                            <text
                              x={node.x}
                              y={node.y + node.radius + 16}
                              textAnchor="middle"
                              className={styles.graphLabel}
                            >
                              {node.label}
                            </text>
                          </g>
                        ))}
                      </svg>
                    </div>
                  </div>
                ) : (
                  <p className={styles.emptyState}>
                    This preview does not contain usable embedding rows for graph rendering.
                  </p>
                )
              ) : (
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
                            <code
                              className={
                                column.name === "source_path"
                                  ? styles.pathCode
                                  : styles.cellCode
                              }
                              title={String(row[column.name] ?? "")}
                            >
                              {formatValue(row[column.name], column.name)}
                            </code>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              )
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
