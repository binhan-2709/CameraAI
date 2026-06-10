const state = {
  apiBase: localStorage.getItem("camai.apiBase") || "http://localhost:8000",
  date: new Date().toISOString().slice(0, 10),
  filter: "all",
  reportRows: [],
  employees: [],
  health: null,
};

const els = {
  apiBase: document.getElementById("api-base"),
  saveApi: document.getElementById("save-api"),
  dateInput: document.getElementById("date-input"),
  refreshBtn: document.getElementById("refresh-btn"),
  exportBtn: document.getElementById("export-btn"),
  apiStatus: document.getElementById("api-status"),
  liveDot: document.getElementById("live-dot"),
  clock: document.getElementById("clock"),
  totalEmployees: document.getElementById("total-employees"),
  presentCount: document.getElementById("present-count"),
  lateCount: document.getElementById("late-count"),
  absentCount: document.getElementById("absent-count"),
  tableCaption: document.getElementById("table-caption"),
  attendanceBody: document.getElementById("attendance-body"),
  employeeList: document.getElementById("employee-list"),
  manualForm: document.getElementById("manual-form"),
  manualEmployee: document.getElementById("manual-employee"),
  manualMessage: document.getElementById("manual-message"),
  sysDetector: document.getElementById("sys-detector"),
  sysEmbedding: document.getElementById("sys-embedding"),
  sysDb: document.getElementById("sys-db"),
  sysPeople: document.getElementById("sys-people"),
  miniDetector: document.getElementById("mini-detector"),
  miniDb: document.getElementById("mini-db"),
  miniEmbedding: document.getElementById("mini-embedding"),
};

function apiUrl(path) {
  return `${state.apiBase.replace(/\/$/, "")}${path}`;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.reason || data.detail || `${response.status} ${response.statusText}`);
  }
  return data;
}

function setApiStatus(ok, text) {
  els.liveDot.className = `dot ${ok ? "ok" : "fail"}`;
  els.apiStatus.textContent = text;
}

function setManualMessage(ok, text) {
  els.manualMessage.className = `manual-message ${ok ? "ok" : "fail"}`;
  els.manualMessage.textContent = text;
}

function formatHours(value) {
  return value ? `${value} giờ` : "-";
}

function statusBadge(row) {
  if (row.status !== "present") {
    return '<span class="badge absent">Vắng</span>';
  }
  if (!row.check_out) {
    return '<span class="badge pending">Chưa checkout</span>';
  }
  if (row.late) {
    return '<span class="badge late">Muộn</span>';
  }
  return '<span class="badge present">Hoàn tất</span>';
}

function filteredRows() {
  return state.reportRows.filter((row) => {
    if (state.filter === "all") return true;
    if (state.filter === "late") return row.late;
    if (state.filter === "checkout_missing") return row.status === "present" && !row.check_out;
    return row.status === state.filter;
  });
}

function renderStats() {
  const present = state.reportRows.filter((row) => row.status === "present").length;
  const late = state.reportRows.filter((row) => row.late).length;
  const absent = state.reportRows.filter((row) => row.status === "absent").length;

  els.totalEmployees.textContent = state.employees.length;
  els.presentCount.textContent = present;
  els.lateCount.textContent = late;
  els.absentCount.textContent = absent;
}

function renderAttendanceTable() {
  const rows = filteredRows();
  els.tableCaption.textContent = `Ngày ${state.date} · ${rows.length} dòng hiển thị`;

  if (!rows.length) {
    els.attendanceBody.innerHTML = '<tr><td class="empty" colspan="7">Không có dữ liệu phù hợp.</td></tr>';
    return;
  }

  els.attendanceBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td><strong>${row.employee_id}</strong></td>
          <td>${row.name || row.employee_id}</td>
          <td>${row.department || "-"}</td>
          <td>${row.check_in || "-"}</td>
          <td>${row.check_out || "-"}</td>
          <td>${formatHours(row.work_hours)}</td>
          <td>${statusBadge(row)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderEmployees() {
  if (!state.employees.length) {
    els.employeeList.innerHTML = '<div class="empty">Chưa có nhân viên active.</div>';
    els.manualEmployee.innerHTML = '<option value="">Chưa có nhân viên</option>';
    return;
  }

  els.employeeList.innerHTML = state.employees
    .map(
      (employee) => `
        <div class="employee-row">
          <strong>${employee.id}</strong>
          <span>${employee.name}</span>
          <span class="muted">${employee.department || "Chưa có phòng ban"}</span>
        </div>
      `,
    )
    .join("");

  const selected = els.manualEmployee.value;
  els.manualEmployee.innerHTML = '<option value="">Chọn nhân viên</option>' + state.employees
    .map((employee) => `<option value="${employee.id}">${employee.id} - ${employee.name}</option>`)
    .join("");
  if (selected) {
    els.manualEmployee.value = selected;
  }
}

function renderSystem() {
  const health = state.health || {};
  const embedding = health.embeddings_enabled ? "Enabled" : "Disabled";
  const people = Array.isArray(health.people) && health.people.length ? health.people.join(", ") : "-";

  els.sysDetector.textContent = health.detector || "-";
  els.sysEmbedding.textContent = embedding;
  els.sysDb.textContent = health.db_size ?? "-";
  els.sysPeople.textContent = people;
  els.miniDetector.textContent = health.detector || "-";
  els.miniDb.textContent = health.db_size ?? "-";
  els.miniEmbedding.textContent = embedding;
}

function renderAll() {
  renderStats();
  renderAttendanceTable();
  renderEmployees();
  renderSystem();
}

async function loadData() {
  setApiStatus(false, "Đang tải dữ liệu...");
  try {
    const [health, employees, report] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/api/employees"),
      fetchJson(`/api/attendance/report?target_date=${state.date}`),
    ]);
    state.health = health;
    state.employees = employees;
    state.reportRows = report.data || [];
    setApiStatus(true, "Backend đang hoạt động");
    renderAll();
  } catch (error) {
    setApiStatus(false, `Không kết nối được backend: ${error.message}`);
    els.attendanceBody.innerHTML = '<tr><td class="empty" colspan="7">Backend chưa sẵn sàng hoặc sai API URL.</td></tr>';
  }
}

async function submitManualAttendance(eventType) {
  const employeeId = els.manualEmployee.value;
  if (!employeeId) {
    setManualMessage(false, "Bạn cần chọn nhân viên trước.");
    return;
  }

  const label = eventType === "check_in" ? "Check-in" : "Check-out";
  setManualMessage(true, `Đang ghi ${label.toLowerCase()}...`);

  try {
    const result = await fetchJson("/api/attendance/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        employee_id: employeeId,
        event_type: eventType,
        camera_id: "MANUAL_UI",
      }),
    });
    setManualMessage(true, `${label} thành công lúc ${new Date(result.timestamp).toLocaleTimeString("vi-VN")}.`);
    await loadData();
  } catch (error) {
    setManualMessage(false, `${label} thất bại: ${translateReason(error.message)}`);
  }
}

function translateReason(reason) {
  const map = {
    already_checked_in: "nhân viên đã check-in hôm nay",
    already_checked_out: "nhân viên đã check-out hôm nay",
    missing_check_in: "chưa có check-in nên chưa thể check-out",
    employee_not_found: "không tìm thấy nhân viên",
    inactive_employee: "nhân viên không còn active",
  };
  return map[reason] || reason;
}

function switchView(viewName) {
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.getElementById(`${viewName}-view`).classList.add("active");

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewName);
  });
}

function initEvents() {
  els.apiBase.value = state.apiBase;
  els.dateInput.value = state.date;

  els.saveApi.addEventListener("click", () => {
    state.apiBase = els.apiBase.value.trim() || "http://localhost:8000";
    localStorage.setItem("camai.apiBase", state.apiBase);
    loadData();
  });

  els.dateInput.addEventListener("change", () => {
    state.date = els.dateInput.value;
    loadData();
  });

  els.refreshBtn.addEventListener("click", loadData);

  els.exportBtn.addEventListener("click", () => {
    window.open(apiUrl(`/api/attendance/report?target_date=${state.date}&format=excel`), "_blank");
  });

  els.manualForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const submitter = event.submitter;
    submitManualAttendance(submitter.dataset.event);
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderAttendanceTable();
    });
  });

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
}

function startClock() {
  const tick = () => {
    els.clock.textContent = new Date().toLocaleTimeString("vi-VN");
  };
  tick();
  setInterval(tick, 1000);
}

initEvents();
startClock();
loadData();
setInterval(loadData, 15000);
