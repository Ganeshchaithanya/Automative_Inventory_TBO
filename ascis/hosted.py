"""Operations website and the two storefront data endpoints for hosted demos."""

from ascis.web import app, book, catalog

app.add_api_route("/api/catalog", catalog, methods=["GET"], include_in_schema=False)
app.add_api_route("/api/bookings", book, methods=["POST"], include_in_schema=False)


@app.get("/healthz", include_in_schema=False)
def health():
    return {"status": "ready"}
