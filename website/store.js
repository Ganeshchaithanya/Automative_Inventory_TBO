const $ = (selector) => document.querySelector(selector);
const amount = new Intl.NumberFormat("en-IN");
const currency = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
let hotels = [];
let catalogSnapshot = "";
let openHotel = null;
let offer = "hotel";
let demoRevision = null;
let spotlightId = null;
let spotlightTimer;
let slideStarted = 0;
const dayMs = 86_400_000;

function node(tag, value, className) {
  const result = document.createElement(tag);
  if (value !== undefined) result.textContent = value;
  if (className) result.className = className;
  return result;
}

function photo(url, alt) {
  const img = node("img");
  img.src = url;
  img.alt = alt;
  img.loading = "lazy";
  return img;
}

function button(text, style, callback) {
  const result = node("button", text, "button " + style);
  result.type = "button";
  result.addEventListener("click", callback);
  return result;
}

function shortDate(value) {
  return new Date(value + "T12:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function showHotel(hotel) {
  openHotel = hotel.id;
  renderHotelDetail(hotel);
  $("#hotel-dialog").showModal();
}

function renderHotelDetail(hotel) {
  const content = node("div", undefined, "dialog-body");
  const title = node("h2", hotel.name);
  title.id = "detail-title";
  const stock = node("p", amount.format(hotel.stock) + " demo room-nights available");
  stock.id = "detail-stock";
  content.append(
    node("p", hotel.area + " · " + hotel.destination), title,
    node("p", hotel.description),
    node("p", "Demo stay dates: " + shortDate(hotel.available_from) + " to " + shortDate(hotel.available_until)),
    stock,
    node("p", offer === "package" ? "Includes the hotel stay and one-way airport transfer per room." : hotel.amenities.join(" · ")),
    bookingForm(hotel, offer),
  );
  const origin = $("#origin").value.trim();
  if (origin) stock.before(node("p", "Trip from " + origin + " to " + hotel.destination));
  $("#hotel-detail").replaceChildren(photo(hotel.hotel_image, "Illustrative photo of " + hotel.name), content);
}

function bookingForm(hotel, product) {
  const bookingRevision = demoRevision ?? 0;
  const form = node("form", undefined, "booking-form");
  function field(name, type, value, min, max) {
    const label = node("label", name);
    const input = node("input");
    Object.assign(input, { type, value, min, max, required: true });
    label.append(input);
    form.append(label);
    return input;
  }
  const firstAvailable = Object.keys(hotel.rooms_by_date).find((day) =>
    hotel.rooms_by_date[day] > 0 && (product === "hotel" || hotel.transfers_by_date[day] > 0));
  const start = field("Check-in", "date", $("#check-in").value || firstAvailable || hotel.available_from,
    hotel.available_from, hotel.available_until);
  const initialOut = $("#check-out").value || new Date(Date.parse(start.value) + dayMs).toISOString().slice(0, 10);
  const end = field("Check-out", "date", initialOut, hotel.available_from, hotel.available_until);
  const rooms = field("Rooms", "number", "1", "1", "12");
  const quote = node("p", undefined, "quote");
  const message = node("p", undefined, "booking-message");
  message.setAttribute("role", "status");
  const submit = node("button", "Create demo booking", "button emerald");
  submit.type = "submit";
  form.append(quote, submit, message);

  function estimate() {
    if (start.disabled) return;
    const current = hotels.find((item) => item.id === hotel.id) || hotel;
    const nights = (Date.parse(end.value) - Date.parse(start.value)) / dayMs;
    const count = Number(rooms.value);
    const valid = form.checkValidity() && nights > 0;
    const total = (nights * hotel.price + (product === "package" ? hotel.transfer_price : 0)) * count;
    const dates = Object.keys(current.rooms_by_date).filter((day) => start.value <= day && day < end.value);
    const roomsAvailable = dates.length === nights && dates.every((day) => current.rooms_by_date[day] >= count);
    const transferAvailable = product === "hotel" || current.transfers_by_date[start.value] >= count;
    if (!valid) quote.textContent = "Choose a valid check-in, check-out and room count.";
    else if (!roomsAvailable) quote.textContent = "There are fewer rooms available for this stay. Try a smaller group or different dates.";
    else if (!transferAvailable) quote.textContent = "There are fewer transfers available on arrival. Try a hotel-only stay or different dates.";
    else quote.textContent = currency.format(total) + " demo total · " + count + (count === 1 ? " room" : " rooms") +
      " × " + nights + (nights === 1 ? " night" : " nights");
    submit.disabled = !valid || !roomsAvailable || !transferAvailable;
  }
  let lastPayload = "";
  let lastId;
  form.addEventListener("input", estimate);
  form.addEventListener("catalogchange", estimate);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (Date.parse(start.value) >= Date.parse(end.value)) return;
    const selection = {
      hotel_id: hotel.id, offer: product, check_in: start.value, check_out: end.value, rooms: Number(rooms.value),
      origin: $("#origin").value.trim(),
      demo_revision: bookingRevision,
    };
    const fingerprint = JSON.stringify(selection);
    if (lastPayload !== fingerprint) {
      lastPayload = fingerprint;
      lastId = crypto.randomUUID();
    }
    submit.disabled = true;
    for (const field of [start, end, rooms]) field.disabled = true;
    try {
      const response = await fetch("/api/bookings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...selection, request_id: lastId }),
      });
      const receipt = await response.json();
      if (!response.ok) throw new Error(typeof receipt.detail === "string" ? receipt.detail : "Check the booking details.");
      message.textContent = "Demo booking " + receipt.reference + " · " + currency.format(receipt.subtotal_inr) + " total.";
      submit.textContent = "Demo booking recorded";
      await refresh();
    } catch (error) {
      message.textContent = error.message;
      for (const field of [start, end, rooms]) field.disabled = false;
      estimate();
      await refresh();
    }
  });
  estimate();
  return form;
}

function card({ hotel, available, featured }) {
  const article = node("article", undefined, "hotel");
  const content = node("div", undefined, "card-body");
  const tags = node("div", undefined, "amenities");
  for (const amenity of offer === "package" ? ["Hotel stay", "One-way airport transfer"] : hotel.amenities) {
    tags.append(node("span", amenity));
  }
  const footer = node("div", undefined, "card-bottom");
  const price = node("div");
  const packageOffer = offer === "package";
  price.append(node("strong", currency.format(hotel.price + (packageOffer ? hotel.transfer_price : 0))),
    node("small", packageOffer ? "from · 1 room, 1 night + transfer" : "per room, per night"));
  footer.append(price, button(packageOffer ? "View package" : "View stay", "outline", () => showHotel(hotel)));
  content.append(
    node("p", hotel.destination, "card-meta"), node("h3", hotel.name),
    node("p", amount.format(available) + " demo room-nights available", "card-meta"),
  );
  if (hotel.event_active) content.append(node("p", hotel.event, "event-promotion"));
  if (featured) {
    content.append(node("p", amount.format(featured) + " room-nights in the event allocation", "allocation"));
  }
  content.append(
    tags, footer,
  );
  article.append(photo(hotel.hotel_image, "Illustrative photo of " + hotel.name), content);
  return article;
}

function renderStays({ scroll = false } = {}) {
  const origin = $("#origin").value.trim();
  const query = $("#query").value.toLowerCase().trim();
  const start = $("#check-in").value;
  const end = $("#check-out").value;
  $("#search-error").textContent = "";
  if ((start || end) && (!start || !end || start >= end)) {
    $("#search-error").textContent = "Enter both dates with check-out after check-in.";
    return;
  }
  const matches = [];
  for (const hotel of hotels) {
    const haystack = [hotel.name, hotel.destination, hotel.area].join(" ").toLowerCase();
    const roomDates = Object.keys(hotel.rooms_by_date).filter((day) => !start || (day >= start && day < end));
    if (!haystack.includes(query) || !hotel.stock) continue;
    if (start && (start < hotel.available_from || end > hotel.available_until ||
      roomDates.some((day) => hotel.rooms_by_date[day] <= 0))) continue;
    if (offer === "package" && start && hotel.transfers_by_date[start] <= 0) continue;
    // With open dates, rank only remaining room-nights on eligible arrival days.
    const eligible = roomDates.filter((day) => hotel.rooms_by_date[day] > 0 &&
      (start || offer === "hotel" || hotel.transfers_by_date[day] > 0));
    if (!eligible.length) continue;
    const available = eligible.reduce((total, day) => total + hotel.rooms_by_date[day], 0);
    const featured = eligible.reduce((total, day) => total + hotel.acquired_by_date[day], 0);
    matches.push({ hotel, available, featured });
  }
  matches.sort((left, right) => right.featured - left.featured);
  const packageOffer = offer === "package";
  $("#stays-title").textContent = packageOffer ? "Hotel + transfer packages" : (query || start ? "Available stays" : "Featured stays");
  $("#offer-description").textContent = packageOffer
    ? "Bundle a stay with a one-way airport transfer on arrival."
    : "Hotel inventory for event travel, city breaks and groups.";
  if (origin && query) {
    $("#offer-description").textContent = "Trip from " + origin + " · Destination: " + $("#query").value.trim();
  }
  $("#result-count").textContent = matches.length + (packageOffer
    ? (matches.length === 1 ? " package" : " packages")
    : (matches.length === 1 ? " hotel" : " hotels"));
  const results = matches.length ? matches.map(card) : [node("p", "No demo stays match these filters. Try a different destination or dates.", "empty")];
  if (!matches.length) {
    results[0].append(document.createTextNode(" "), button("Clear filters", "outline", () => {
      $("#search").reset();
      renderStays();
    }));
  }
  $("#hotel-grid").replaceChildren(...results);
  if (scroll) $("#stays").scrollIntoView({ block: "start" });
}

function renderDestinations() {
  $("#destination-grid").replaceChildren(...hotels.map((hotel) => {
    const [city, country] = hotel.destination.split(", ");
    const tile = node("button", undefined, "destination");
    tile.type = "button";
    const label = node("div");
    label.append(node("strong", city), node("span", country));
    tile.append(photo(hotel.city_image, ""), label);
    tile.addEventListener("click", () => {
      $("#query").value = city;
      renderStays({ scroll: true });
    });
    return tile;
  }));
}

function spotlightHotels() {
  return hotels.filter((hotel) => hotel.promotion_seconds > 0).sort((left, right) => left.id.localeCompare(right.id));
}

function advanceSpotlight() {
  const slides = spotlightHotels();
  if (!slides.length) return;
  const index = slides.findIndex((hotel) => hotel.id === spotlightId);
  spotlightId = slides[(index + 1) % slides.length].id;
  slideStarted = performance.now();
  renderSpotlight();
}

function scheduleSpotlight() {
  clearTimeout(spotlightTimer);
  const slides = spotlightHotels();
  const featured = slides.find((hotel) => hotel.id === spotlightId);
  if (!featured || slides.length < 2 || document.hidden ||
      $("#spotlight-slide").matches(":hover, :focus-within")) return;
  spotlightTimer = setTimeout(
    advanceSpotlight, Math.max(0, featured.promotion_seconds * 1000 - (performance.now() - slideStarted)),
  );
}

function renderSpotlight() {
  const section = $("#spotlight");
  const slot = $("#spotlight-slide");
  const slides = spotlightHotels();
  let featured = slides.find((hotel) => hotel.id === spotlightId);
  if (!featured && slides.length) {
    featured = slides.reduce((longest, hotel) => hotel.promotion_seconds > longest.promotion_seconds ? hotel : longest);
    spotlightId = featured.id;
    slideStarted = performance.now();
  }
  section.hidden = !featured;
  if (!featured) {
    spotlightId = null;
    clearTimeout(spotlightTimer);
    slot.replaceChildren();
    return;
  }
  const promotion = node("div", undefined, "spotlight");
  if (slot.firstElementChild?.dataset.event !== featured.id) promotion.classList.add("spotlight-enter");
  promotion.dataset.event = featured.id;
  const copy = node("div");
  copy.append(
    node("p", featured.destination, "eyebrow"),
    node("h2", featured.event),
    node("p", featured.name + " · " + amount.format(featured.featured_stock || featured.stock) +
      (featured.featured_stock ? " room-nights in the approved event allocation" : " demo room-nights available")),
  );
  promotion.append(photo(featured.city_image, ""), copy, button("Explore event stay", "orange", () => {
    $("#query").value = featured.name;
    renderStays({ scroll: true });
  }));
  if (slot.firstElementChild?.dataset.event === featured.id) {
    slot.querySelector(".spotlight-availability").textContent = copy.lastElementChild.textContent;
  } else {
    copy.lastElementChild.classList.add("spotlight-availability");
    slot.replaceChildren(promotion);
  }
  scheduleSpotlight();
}

$("#spotlight-slide").addEventListener("pointerenter", scheduleSpotlight);
$("#spotlight-slide").addEventListener("focusin", scheduleSpotlight);
for (const event of ["pointerleave", "focusout"]) {
  $("#spotlight-slide").addEventListener(event, () => {
    slideStarted = performance.now();
    requestAnimationFrame(scheduleSpotlight);
  });
}
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) slideStarted = performance.now();
  scheduleSpotlight();
});

$("#search").addEventListener("submit", (event) => {
  event.preventDefault();
  renderStays({ scroll: true });
});
for (const control of document.querySelectorAll("[data-offer]")) {
  control.addEventListener("click", () => {
    offer = control.dataset.offer;
    for (const tab of document.querySelectorAll("button[data-offer]")) {
      tab.setAttribute("aria-pressed", String(tab.dataset.offer === offer));
    }
    renderStays();
  });
}

async function refresh() {
  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error("The hotel collection is unavailable.");
    const data = await response.json();
    const revision = Number(response.headers.get("X-ASCIS-Demo-Revision"));
    if (demoRevision !== null && revision < demoRevision) return;
    if (demoRevision !== null && revision > demoRevision) {
      if ($("#hotel-dialog").open) $("#hotel-dialog").close();
      $("#search").reset();
      offer = "hotel";
      for (const tab of document.querySelectorAll("button[data-offer]")) {
        tab.setAttribute("aria-pressed", String(tab.dataset.offer === offer));
      }
      openHotel = null;
      spotlightId = null;
      catalogSnapshot = "";
    }
    demoRevision = revision;
    const snapshot = JSON.stringify(data);
    if (snapshot === catalogSnapshot) return;
    catalogSnapshot = snapshot;
    hotels = data;
    renderStays();
    renderDestinations();
    renderSpotlight();
    if ($("#hotel-dialog").open && openHotel) {
      const hotel = hotels.find((item) => item.id === openHotel);
      if (hotel) {
        $("#detail-stock").textContent = amount.format(hotel.stock) + " demo room-nights available";
        $("#hotel-detail .booking-form").dispatchEvent(new Event("catalogchange"));
      }
    }
  } catch {
    if (!hotels.length) {
      $("#hotel-grid").replaceChildren(node("p", "The hotel collection is temporarily unavailable.", "empty"));
    }
  }
}

await refresh();
window.addEventListener("focus", refresh);
setInterval(refresh, 2000);
