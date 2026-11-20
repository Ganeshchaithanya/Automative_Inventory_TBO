const methods = { catalog: "GET", bookings: "POST" };

export default async function handler(request, response) {
  const resource = request.query.resource;
  if (typeof resource !== "string" || !Object.hasOwn(methods, resource)) {
    return response.status(404).json({ detail: "Unknown storefront resource." });
  }
  if (request.method !== methods[resource]) {
    response.setHeader("Allow", methods[resource]);
    return response.status(405).json({ detail: "Method not allowed." });
  }
  if (!process.env.ASCIS_OPERATIONS_URL) {
    return response.status(503).json({ detail: "The storefront backend is not configured." });
  }
  try {
    const origin = new URL(process.env.ASCIS_OPERATIONS_URL);
    const booking = resource === "bookings";
    const upstream = await fetch(new URL("/api/" + resource, origin), {
      method: request.method,
      headers: booking ? { "Content-Type": "application/json" } : undefined,
      body: booking ? typeof request.body === "string" ? request.body : JSON.stringify(request.body) : undefined,
      redirect: "error",
      signal: AbortSignal.timeout(45000),
    });
    for (const header of ["content-type", "x-ascis-demo-revision"]) {
      const value = upstream.headers.get(header);
      if (value) response.setHeader(header, value);
    }
    response.setHeader("Cache-Control", "no-store");
    response.status(upstream.status).send(await upstream.text());
  } catch {
    response.status(502).json({ detail: "The storefront backend is temporarily unavailable." });
  }
}
