## Postgres Viewer

Paste a PostgreSQL connection URL into the app, browse tables and views, and preview up to 100 rows without putting the connection string in the page URL.

## Run locally

Install dependencies and start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000), paste a URL like the one below, and connect:

```text
postgresql://user:password@host:5432/database?sslmode=require
```

## What it does

- Lists tables, partitioned tables, views, materialized views, and foreign tables
- Shows column counts for each object
- Previews up to 100 rows for the selected object
- Compresses long cell values like large text fields, JSON, and vectors into compact previews
- Keeps important path-like fields such as `source_path` visible
- Supports common hosted Postgres URLs that require `sslmode=require`

## Notes

- The connection string is submitted to the server with each action instead of being stored in the browser URL.
- This is a read-only browser. It only runs inspection queries and `select * ... limit 100` for previews.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
