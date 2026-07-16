import { createServer } from "vite";

const port = 4173;

export default async function globalSetup(): Promise<() => Promise<void>> {
  const server = await createServer({
    server: {
      host: "127.0.0.1",
      port,
      strictPort: true,
    },
  });
  await server.listen();

  return async () => {
    await server.close();
  };
}
