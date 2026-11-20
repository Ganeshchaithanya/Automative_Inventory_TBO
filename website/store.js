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
const retryBookings = new Map();
const cartStorageKey = "avoflare-booking-cart";
let cart = { revision: null, items: [], pending: null };
try { cart = JSON.parse(sessionStorage.getItem(cartStorageKey)) || cart; } catch { /* Start with an empty cart. */ }
let cartSending = false;
let cartReceipt;

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
  return new Date(value + "T00:00:00Z").toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
  });
}

function showHotel(hotel, product = offer) {
  openHotel = hotel.id;
  renderHotelDetail(hotel, product);
  $("#hotel-dialog").showModal();
}

function renderHotelDetail(hotel, product) {
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
    node("p", product === "package" ? "Includes the hotel stay and one-way airport transfer per room." : hotel.amenities.join(" · ")),
    bookingForm(hotel, product),
  );
  const origin = $("#origin").value.trim();
  if (origin) stock.before(node("p", "Trip from " + origin + " to " + hotel.destination));
  $("#hotel-detail").replaceChildren(photo(hotel.hotel_image, "Illustrative photo of " + hotel.name), content);
}

function pendingBooking(key) {
  if (retryBookings.has(key)) return retryBookings.get(key);
  try { return JSON.parse(sessionStorage.getItem(key)); } catch { return null; }
}

function renderBookingRecovery() {
  const panel = $("#booking-recovery");
  const entries = [];
  for (const hotel of hotels) {
    for (const product of ["hotel", "package"]) {
      if (!pendingBooking(`booking:${demoRevision}:${hotel.id}:${product}`)) continue;
      entries.push(button("Recover " + hotel.name + (product === "package" ? " package" : " stay"), "outline",
        () => showHotel(hotel, product)));
    }
  }
  panel.hidden = !entries.length;
  panel.replaceChildren(...entries);
}

function groupQuote(hotel, item) {
  const dates = Object.keys(hotel.rooms_by_date).filter((day) => item.check_in <= day && day < item.check_out);
  const nights = (Date.parse(item.check_out) - Date.parse(item.check_in)) / dayMs;
  if (!(nights > 0) || dates.length !== nights) return { capacity: 0, dates: [], total: 0 };
  const roomLimit = Math.min(...dates.map((day) => hotel.rooms_by_date[day]));
  const transfers = item.offer === "package";
  const capacity = transfers ? Math.min(roomLimit, hotel.transfers_by_date[item.check_in] ?? 0) : roomLimit;
  return { capacity, roomLimit, dates,
    total: (nights * hotel.price + (transfers ? hotel.transfer_price : 0)) * item.rooms };
}

function saveCart() {
  try { sessionStorage.setItem(cartStorageKey, JSON.stringify(cart)); } catch { /* Keep the current tab usable. */ }
  renderCart();
}

function openCart() {
  renderCart();
  if (!$("#cart-dialog").open) $("#cart-dialog").showModal();
}

function renderCart() {
  const list = $("#cart-items");
  const totalLabel = $("#cart-total");
  const checkout = $("#cart-checkout");
  $("#open-cart").textContent = "Cart (" + cart.items.length + ")";
  const editable = !cart.pending && !cartSending;
  checkout.textContent = cartSending ? "Checking inventory…" :
    cart.pending ? "Retry cart confirmation" : "Confirm demo bookings";
  checkout.hidden = cart.items.length === 0;
  const available = new Map(hotels.map((hotel) => [hotel.id, {
    ...hotel, rooms_by_date: { ...hotel.rooms_by_date }, transfers_by_date: { ...hotel.transfers_by_date },
  }]));
  let total = 0;
  let valid = cart.items.length > 0;
  const rows = cart.items.map((item, index) => {
    const hotel = available.get(item.hotel_id);
    const quote = hotel ? groupQuote(hotel, item) : { capacity: 0, total: 0 };
    const fits = item.rooms <= quote.capacity;
    valid &&= fits;
    total += quote.total;
    if (fits) {
      for (const day of quote.dates) hotel.rooms_by_date[day] -= item.rooms;
      if (item.offer === "package") hotel.transfers_by_date[item.check_in] -= item.rooms;
    }
    const title = hotel?.name || "Unavailable property";
    const row = node("article", undefined, "cart-line");
    const quantity = node("input");
    Object.assign(quantity, { type: "number", min: "1", step: "1", value: String(item.rooms), disabled: !editable });
    quantity.setAttribute("aria-label", "Rooms for cart item " + (index + 1) + ", " + title);
    quantity.addEventListener("change", () => {
      if (!Number.isSafeInteger(quantity.valueAsNumber) || quantity.valueAsNumber < 1) {
        quantity.value = String(item.rooms);
        return;
      }
      item.rooms = quantity.valueAsNumber;
      $("#cart-message").textContent = "";
      saveCart();
    });
    const remove = button("Remove", "outline", () => {
      cart.items.splice(index, 1);
      $("#cart-message").textContent = "";
      saveCart();
    });
    remove.disabled = !editable;
    row.append(
      node("h3", (index + 1) + ". " + title),
      node("p", (item.offer === "package" ? "Hotel + transfer" : "Hotel only") + " · " +
        shortDate(item.check_in) + " to " + shortDate(item.check_out)),
      node("p", editable && !fits ? "Only " + amount.format(quote.capacity) + " rooms fit after earlier cart items." : currency.format(quote.total),
        editable && !fits ? "cart-warning" : undefined),
      quantity, remove,
    );
    return row;
  });
  if (cartReceipt && !rows.length) {
    rows.push(node("strong", "Cart confirmed · " + cartReceipt.reference));
    for (const booking of cartReceipt.bookings) {
      rows.push(node("p", booking.hotel + " · " + booking.reference + " · " + amount.format(booking.rooms) + " rooms · " +
        shortDate(booking.nights[0]) + " to " + shortDate(booking.check_out)));
    }
    total = cartReceipt.subtotal_inr;
  }
  list.replaceChildren(...(rows.length ? rows : [node("p", "Add hotels or packages to build a group booking.")]));
  totalLabel.textContent = rows.length ? (cartReceipt && !cart.items.length ? "Confirmed demo total " : "Estimated demo total ") +
    currency.format(total) : "";
  checkout.disabled = cartSending || (!cart.pending && (!valid || demoRevision === null));
}

async function confirmCart() {
  if (cartSending || $("#cart-checkout").disabled) return;
  cart.pending ||= { request_id: crypto.randomUUID(), demo_revision: cart.revision, items: cart.items.map((item) => ({ ...item })) };
  const request = cart.pending;
  cartSending = true;
  $("#cart-message").textContent = "";
  saveCart();
  try {
    const response = await fetch("/api/booking-carts", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
    });
    const result = await response.json();
    if (cart.pending !== request) return;
    if (!response.ok) {
      if (response.status >= 400 && response.status < 500) cart.pending = null;
      throw new Error(typeof result.detail === "string" ? result.detail : "Check your cart quantities and dates.");
    }
    if (!result.reference || !result.bookings?.length) throw new Error("Cart confirmation is unavailable.");
    cart.items = [];
    cart.pending = null;
    cartReceipt = result;
    $("#cart-message").textContent = "All demo bookings confirmed.";
  } catch (error) {
    if (cart.revision === request.demo_revision) {
      $("#cart-message").textContent = cart.pending ? "Confirmation is uncertain. Retry this cart to recover the same result." : error.message;
    }
  } finally {
    cartSending = false;
    saveCart();
    await refresh();
  }
}

function bookingForm(hotel, product) {
  const bookingRevision = demoRevision ?? 0;
  const retryKey = `booking:${bookingRevision}:${hotel.id}:${product}`;
  let pending = pendingBooking(retryKey);
  function savePending(selection) {
    pending = selection;
    if (selection) retryBookings.set(retryKey, selection);
    else retryBookings.delete(retryKey);
    try {
      if (selection) sessionStorage.setItem(retryKey, JSON.stringify(selection));
      else sessionStorage.removeItem(retryKey);
    } catch { /* Storage is optional for same-page retries. */ }
    renderBookingRecovery();
  }
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
  const start = field("Check-in", "date", pending?.check_in || $("#check-in").value || firstAvailable || hotel.available_from,
    hotel.available_from, hotel.available_until);
  const initialOut = pending?.check_out || $("#check-out").value || new Date(Date.parse(start.value) + dayMs).toISOString().slice(0, 10);
  const end = field("Check-out", "date", initialOut, hotel.available_from, hotel.available_until);
  const rooms = field("Rooms", "number", String(pending?.rooms || 1), "1", "");
  const quote = node("p", undefined, "quote");
  const message = node("p", undefined, "booking-message");
  message.setAttribute("role", "status");
  const submit = node("button", "Add to cart", "button emerald");
  submit.type = "submit";
  const receiptPanel = node("div", undefined, "booking-receipt");
  receiptPanel.hidden = true;
  form.append(quote, submit, message, receiptPanel);
  let sending = false;
  let recorded = false;

  function estimate() {
    if (sending || recorded) return;
    for (const field of [start, end, rooms]) field.disabled = Boolean(pending);
    if (pending) {
      quote.textContent = "A confirmation is pending for these dates. Retry to get the result of the same request.";
      submit.textContent = "Retry booking confirmation";
      submit.disabled = false;
      return;
    }
    submit.textContent = "Add to cart";
    const current = hotels.find((item) => item.id === hotel.id) || hotel;
    const count = Number(rooms.value);
    const stay = groupQuote(current, { check_in: start.value, check_out: end.value, rooms: count, offer: product });
    const valid = form.checkValidity() && stay.dates.length > 0 && Number.isSafeInteger(count);
    if (cart.pending) quote.textContent = "Recover your pending cart confirmation before adding more stays.";
    else if (!valid) quote.textContent = "Choose a valid check-in, check-out and room count.";
    else quote.textContent = amount.format(stay.capacity) + " rooms available for the full stay" +
      (product === "package" && stay.capacity < stay.roomLimit ? " with arrival transfers" : "") +
      (count <= stay.capacity ? " · " + currency.format(stay.total) + " estimated total" : ". Adjust the group size or dates.");
    submit.disabled = !valid || count > stay.capacity || Boolean(cart.pending);
  }
  start.addEventListener("input", () => {
    if (start.value && (!end.value || end.value <= start.value) && start.value < hotel.available_until) {
      end.value = new Date(Date.parse(start.value) + dayMs).toISOString().slice(0, 10);
    }
  });
  form.addEventListener("input", estimate);
  form.addEventListener("catalogchange", estimate);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (sending || recorded || (!pending && submit.disabled)) return;
    if (!pending) {
      if (cart.pending) return;
      cart.items.push({
        hotel_id: hotel.id, offer: product, check_in: start.value, check_out: end.value,
        rooms: Number(rooms.value), origin: $("#origin").value.trim(),
      });
      cartReceipt = undefined;
      $("#cart-message").textContent = "";
      saveCart();
      $("#hotel-dialog").close();
      openCart();
      return;
    }
    const selection = pending;
    savePending(selection);
    sending = true;
    message.textContent = "";
    submit.textContent = "Checking availability…";
    submit.disabled = true;
    for (const field of [start, end, rooms]) field.disabled = true;
    try {
      const response = await fetch("/api/bookings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selection),
      });
      const receipt = await response.json();
      if (!response.ok) {
        if (response.status >= 400 && response.status < 500) {
          savePending(null);
          throw new Error(typeof receipt.detail === "string" ? receipt.detail : "Check the booking details and try again.");
        }
        throw new Error("Booking confirmation is temporarily unavailable. Retry the same request.");
      }
      if (!receipt.reference || !receipt.nights?.length) throw new Error("Retry to recover the booking reference.");
      savePending(null);
      recorded = true;
      message.textContent = "Demo booking confirmed. Reference " + receipt.reference + ".";
      quote.hidden = true;
      receiptPanel.append(
        node("strong", receipt.hotel + " · " + (receipt.offer === "package" ? "Hotel + transfer" : "Hotel only")),
        node("p", shortDate(receipt.nights[0]) + " to " + shortDate(receipt.check_out) + " · " +
          amount.format(receipt.room_nights) + " room-nights"),
        node("p", currency.format(receipt.subtotal_inr) + " demo total"),
      );
      receiptPanel.hidden = false;
      submit.textContent = "Demo booking recorded";
      await refresh();
    } catch (error) {
      message.textContent = pending ? "The booking result is uncertain. Retry to recover the same confirmation." : error.message;
    } finally {
      sending = false;
      estimate();
      if (!recorded) await refresh();
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
    $("#hotel-grid").replaceChildren();
    $("#result-count").textContent = "";
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
$("#open-cart").addEventListener("click", openCart);
$("#cart-checkout").addEventListener("click", confirmCart);
for (const control of document.querySelectorAll("[data-offer]")) {
  control.addEventListener("click", () => {
    offer = control.dataset.offer;
    for (const tab of document.querySelectorAll("button[data-offer]")) {
      tab.setAttribute("aria-pressed", String(tab.dataset.offer === offer));
    }
    renderStays();
  });
}

let catalogRequest = 0;
let appliedCatalog = 0;
async function refresh() {
  const request = ++catalogRequest;
  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error("The hotel collection is unavailable.");
    const data = await response.json();
    if (request < appliedCatalog) return;
    const revision = Number(response.headers.get("X-ASCIS-Demo-Revision"));
    if (demoRevision !== null && revision < demoRevision) return;
    appliedCatalog = request;
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
    if (cart.revision !== revision) {
      cart = { revision, items: [], pending: null };
      cartReceipt = undefined;
      $("#cart-message").textContent = "";
      saveCart();
    }
    const snapshot = JSON.stringify(data);
    if (snapshot === catalogSnapshot) return;
    catalogSnapshot = snapshot;
    hotels = data;
    renderStays();
    renderBookingRecovery();
    renderCart();
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
