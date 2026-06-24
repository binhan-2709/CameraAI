let savedApiBase = localStorage.getItem("camai.apiBase");
if (savedApiBase === "http://localhost:8000") {
  savedApiBase = "http://127.0.0.1:8000";
  localStorage.setItem("camai.apiBase", savedApiBase);
}

const state = {
  apiBase: savedApiBase || "http://127.0.0.1:8000",
  date: new Date().toISOString().slice(0, 10),
  filter: "all",
  reportRows: [],
  employees: [],
  health: null,
  employeeSearchQuery: "",
  recentLogs: [],
  logs: [],
  logsSearchQuery: "",
  logsFilterType: "all",
};

const registration = {
  mode: "camera",
  stream: null,
  photos: [],
  lastCaptureTime: 0,
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
  registerForm: document.getElementById("register-form"),
  regEmployeeId: document.getElementById("reg-employee-id"),
  regName: document.getElementById("reg-name"),
  regDepartment: document.getElementById("reg-department"),
  cameraMode: document.getElementById("camera-mode"),
  uploadMode: document.getElementById("upload-mode"),
  cameraPreview: document.getElementById("camera-preview"),
  startCameraBtn: document.getElementById("start-camera-btn"),
  stopCameraBtn: document.getElementById("stop-camera-btn"),
  captureBtn: document.getElementById("capture-btn"),
  uploadTriggerBtn: document.getElementById("upload-trigger-btn"),
  fileInput: document.getElementById("file-input"),
  photosGrid: document.getElementById("photos-grid"),
  photoCount: document.getElementById("photo-count"),
  registerSubmit: document.getElementById("register-submit"),
  registerMessage: document.getElementById("register-message"),
  modeBtns: document.querySelectorAll(".mode-btn"),
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

function setRegisterMessage(ok, text) {
  els.registerMessage.className = `register-message ${ok ? "ok" : "fail"}`;
  els.registerMessage.textContent = text;
}

function updatePhotoCount() {
  const minPhotos = 3;
  const photoCount = registration.photos.length;
  els.photoCount.textContent = `Ảnh đã chọn: ${photoCount}/${minPhotos}+`;
  els.registerSubmit.disabled = photoCount < minPhotos;
}

function renderPhotos() {
  els.photosGrid.innerHTML = registration.photos
    .map(
      (photo, index) => `
        <div class="photo-item">
          <img src="${photo.dataUrl}" alt="Photo ${index + 1}">
          <button class="delete-btn" type="button" data-index="${index}">✕</button>
        </div>
      `,
    )
    .join("");

  document.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const index = parseInt(btn.dataset.index);
      registration.photos.splice(index, 1);
      renderPhotos();
      updatePhotoCount();
    });
  });

  updatePhotoCount();
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    registration.stream = stream;
    els.cameraPreview.srcObject = stream;
    els.cameraPreview.style.display = "block";
    els.startCameraBtn.style.display = "none";
    els.stopCameraBtn.style.display = "block";
    els.captureBtn.style.display = "block";

    try {
      await els.cameraPreview.play();
    } catch (e) {
      console.warn("Autoplay was prevented, waiting for user interaction:", e);
    }

    setRegisterMessage(true, "Camera sẵn sàng. Ấn 'Chụp ảnh' để chụp.");
  } catch (error) {
    setRegisterMessage(false, `Lỗi truy cập camera: ${error.message}`);
  }
}

function stopCamera() {
  if (registration.stream) {
    registration.stream.getTracks().forEach((track) => track.stop());
    registration.stream = null;
  }
  els.cameraPreview.srcObject = null;
  els.cameraPreview.style.display = "none";
  els.startCameraBtn.style.display = "block";
  els.stopCameraBtn.style.display = "none";
  els.captureBtn.style.display = "none";
}

function capturePhoto() {
  try {
    const now = Date.now();
    if (now - registration.lastCaptureTime < 500) {
      setRegisterMessage(false, "Chụp quá nhanh. Vui lòng chờ.");
      return;
    }
    registration.lastCaptureTime = now;

    if (!registration.stream) {
      setRegisterMessage(false, "Camera chưa được kích hoạt hoặc không hoạt động.");
      return;
    }

    const canvas = document.createElement("canvas");
    const width = els.cameraPreview.videoWidth || els.cameraPreview.clientWidth || 640;
    const height = els.cameraPreview.videoHeight || els.cameraPreview.clientHeight || 480;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(els.cameraPreview, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg");

    canvas.toBlob((blob) => {
      if (!blob) {
        setRegisterMessage(false, "Lỗi khi chuyển đổi ảnh chụp từ camera stream.");
        return;
      }
      try {
        const file = new File([blob], `capture-${registration.photos.length + 1}.jpg`, { type: "image/jpeg" });
        registration.photos.push({ dataUrl, file });
        renderPhotos();
        setRegisterMessage(true, `Đã chụp được ${registration.photos.length} ảnh. ${registration.photos.length >= 3 ? "Có thể đăng ký!" : ""}`);
      } catch (err) {
        setRegisterMessage(false, `Lỗi tạo file ảnh: ${err.message}`);
      }
    }, "image/jpeg");
  } catch (error) {
    setRegisterMessage(false, `Lỗi chụp ảnh: ${error.message}`);
  }
}

function handleFileUpload(files) {
  for (const file of files) {
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setRegisterMessage(false, `File ${file.name} không được hỗ trợ. Chỉ JPG, PNG, WEBP.`);
      continue;
    }

    if (file.size > 8 * 1024 * 1024) {
      setRegisterMessage(false, `File ${file.name} quá lớn (>8MB).`);
      continue;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      registration.photos.push({ dataUrl: e.target.result, file });
      renderPhotos();
    };
    reader.readAsDataURL(file);
  }
}

async function submitRegistration() {
  const employeeId = els.regEmployeeId.value.trim();
  const name = els.regName.value.trim();
  const department = els.regDepartment.value.trim();

  if (!employeeId || !name) {
    setRegisterMessage(false, "Mã nhân viên và Họ tên bắt buộc.");
    return;
  }

  if (registration.photos.length < 3) {
    setRegisterMessage(false, "Phải có ít nhất 3 ảnh.");
    return;
  }

  const formData = new FormData();
  formData.append("employee_id", employeeId);
  formData.append("name", name);
  formData.append("department", department);
  registration.photos.forEach((photo) => {
    formData.append("files", photo.file);
  });

  els.registerSubmit.disabled = true;
  setRegisterMessage(true, "Đang đăng ký nhân viên...");

  try {
    const result = await fetch(apiUrl("/api/employees/register"), {
      method: "POST",
      body: formData,
    }).then((res) => {
      if (!res.ok) {
        return res.json().then((data) => {
          throw new Error(data.detail || `${res.status}`);
        });
      }
      return res.json();
    });

    setRegisterMessage(true, `✓ Đã thêm ${result.employee_id} thành công! ${result.images_saved} ảnh, ${result.embeddings_added} embedding.`);

    els.regEmployeeId.value = "";
    els.regName.value = "";
    els.regDepartment.value = "";
    els.fileInput.value = "";
    registration.photos = [];
    renderPhotos();
    // Keep camera running (do not call stopCamera) for live registration

    await new Promise((resolve) => setTimeout(resolve, 2000));
    await loadData();
  } catch (error) {
    setRegisterMessage(false, `Lỗi đăng ký: ${error.message}`);
    els.registerSubmit.disabled = false;
  }

  updatePhotoCount();
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
  const query = (state.employeeSearchQuery || "").toLowerCase().trim();
  const filteredEmployees = state.employees.filter((emp) => 
    (emp.id || "").toLowerCase().includes(query) || 
    (emp.name || "").toLowerCase().includes(query) || 
    (emp.department || "").toLowerCase().includes(query)
  );

  if (!filteredEmployees.length) {
    els.employeeList.innerHTML = '<div class="empty">Không tìm thấy nhân viên phù hợp.</div>';
    els.manualEmployee.innerHTML = '<option value="">Chưa có nhân viên</option>';
    return;
  }

  els.employeeList.innerHTML = filteredEmployees
    .map(
      (employee) => {
        const att = state.reportRows.find((row) => row.employee_id === employee.id);
        let statusBadge = '<span class="badge absent">Vắng</span>';
        if (att) {
          if (att.status === "present") {
            if (att.late) {
              statusBadge = '<span class="badge late">Đi muộn</span>';
            } else {
              statusBadge = '<span class="badge present">Có mặt</span>';
            }
          } else if (att.status === "pending") {
            statusBadge = '<span class="badge pending">Chưa checkout</span>';
          }
        }
        return `
          <div class="employee-row">
            <strong>${employee.id}</strong>
            <span>${employee.name}</span>
            <span class="muted">${employee.department || "Chưa có phòng ban"}</span>
            <div class="employee-status">${statusBadge}</div>
            <button class="delete-employee-btn" type="button" data-id="${employee.id}" title="Xóa nhân viên">✕</button>
          </div>
        `;
      },
    )
    .join("");

  // Bind delete events
  document.querySelectorAll(".delete-employee-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      const employeeId = btn.dataset.id;
      if (confirm(`Bạn có chắc chắn muốn xóa nhân viên ${employeeId} khỏi hệ thống?`)) {
        try {
          btn.disabled = true;
          await fetchJson(`/api/employees/${employeeId}`, { method: "DELETE" });
          await loadData();
        } catch (error) {
          alert(`Lỗi khi xóa nhân viên: ${error.message}`);
          btn.disabled = false;
        }
      }
    });
  });

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

  if (els.sysDetector) els.sysDetector.textContent = health.detector || "-";
  if (els.sysEmbedding) els.sysEmbedding.textContent = embedding;
  if (els.sysDb) els.sysDb.textContent = health.db_size ?? "-";
  if (els.sysPeople) els.sysPeople.textContent = people;
  if (els.miniDetector) els.miniDetector.textContent = health.detector || "-";
  if (els.miniDb) els.miniDb.textContent = health.db_size ?? "-";
  if (els.miniEmbedding) els.miniEmbedding.textContent = embedding;
}

async function loadRecentLogs() {
  try {
    const logs = await fetchJson("/api/attendance/logs?limit=10");
    state.recentLogs = logs || [];
    renderRecentLogs();
  } catch (error) {
    console.error("Lỗi khi tải nhật ký:", error);
    const container = document.getElementById("recent-logs-list");
    if (container) {
      container.innerHTML = '<div class="empty">Không thể tải nhật ký.</div>';
    }
  }
}

function renderRecentLogs() {
  const container = document.getElementById("recent-logs-list");
  if (!container) return;

  if (!state.recentLogs.length) {
    container.innerHTML = '<div class="empty">Chưa có nhật ký chấm công hôm nay.</div>';
    return;
  }

  container.innerHTML = state.recentLogs
    .map((log) => {
      const timeStr = log.time || new Date(log.timestamp).toLocaleTimeString("vi-VN");
      const eventLabel = log.event_type === "check_in" ? "Check-in" : "Check-out";
      const confidencePct = Math.round(log.confidence * 100);
      
      const isReal = log.is_real_face !== false;
      const itemStyle = isReal ? "" : ' style="background-color: #fef2f2; border-left: 4px solid var(--red); padding-left: 8px;"';
      const nameHtml = isReal ? `${log.name || log.employee_id}` : `${log.name || log.employee_id} <span style="color:var(--red); font-size:10px; font-weight:bold;">[GIẢ MẠO]</span>`;

      return `
        <div class="log-item"${itemStyle}>
          <div class="log-info">
            <span class="log-name">${nameHtml}</span>
            <span class="log-dept">${log.department || "Nhân viên"} · ${eventLabel}</span>
          </div>
          <div class="log-time-badge">
            <span class="log-time">${timeStr}</span>
            <span class="log-confidence">Tin cậy: ${confidencePct}%</span>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderAll() {
  renderStats();
  renderAttendanceTable();
  renderEmployees();
  renderSystem();
  renderRecentLogs();
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
    loadRecentLogs();
  } catch (error) {
    console.error("Lỗi kết nối hoặc xử lý dữ liệu từ backend:", error);
    setApiStatus(false, `Không kết nối được backend: ${error.message}`);
    els.attendanceBody.innerHTML = '<tr><td class="empty" colspan="7">Backend chưa sẵn sàng hoặc sai API URL. Bạn có thể cấu hình địa chỉ API tại tab <strong>Hệ thống</strong>.</td></tr>';
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

async function loadSettings() {
  const msgEl = document.getElementById("settings-message");
  if (msgEl) msgEl.textContent = "";
  try {
    const settings = await fetchJson("/api/settings");
    const lateEl = document.getElementById("set-late-threshold");
    const cooldownEl = document.getElementById("set-cooldown");
    const faceEl = document.getElementById("set-face-threshold");
    if (lateEl) lateEl.value = settings.late_threshold || "08:30";
    if (cooldownEl) cooldownEl.value = settings.cooldown_minutes ?? 5;
    if (faceEl) faceEl.value = settings.face_threshold ?? 0.5;
  } catch (error) {
    console.error("Lỗi khi tải cài đặt:", error);
    if (msgEl) {
      msgEl.className = "fail";
      msgEl.style.color = "var(--red)";
      msgEl.textContent = "Không thể tải cấu hình từ server.";
    }
  }
}

async function saveSettings(e) {
  e.preventDefault();
  const msgEl = document.getElementById("settings-message");
  if (msgEl) {
    msgEl.className = "";
    msgEl.style.color = "";
    msgEl.textContent = "Đang lưu cấu hình...";
  }
  const lateThreshold = document.getElementById("set-late-threshold").value.trim();
  const cooldownMinutes = parseInt(document.getElementById("set-cooldown").value);
  const faceThreshold = parseFloat(document.getElementById("set-face-threshold").value);

  try {
    await fetchJson("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        late_threshold: lateThreshold,
        cooldown_minutes: cooldownMinutes,
        face_threshold: faceThreshold,
      }),
    });
    if (msgEl) {
      msgEl.className = "ok";
      msgEl.style.color = "var(--green)";
      msgEl.textContent = "✓ Lưu cài đặt thành công!";
    }
    setTimeout(() => {
      if (msgEl) msgEl.textContent = "";
    }, 3000);
  } catch (error) {
    if (msgEl) {
      msgEl.className = "fail";
      msgEl.style.color = "var(--red)";
      msgEl.textContent = `Lỗi: ${error.message}`;
    }
  }
}

async function loadAllLogs() {
  const tbody = document.getElementById("logs-table-body");
  if (tbody && state.logs.length === 0) {
    tbody.innerHTML = '<tr><td class="empty" colspan="9">Đang tải nhật ký...</td></tr>';
  }
  try {
    const logs = await fetchJson("/api/attendance/logs?limit=100");
    state.logs = logs || [];
    renderAllLogsTable();
  } catch (error) {
    console.error("Lỗi khi tải nhật ký chi tiết:", error);
    if (tbody) {
      tbody.innerHTML = `<tr><td class="empty" colspan="9">Không thể tải nhật ký từ server: ${error.message}</td></tr>`;
    }
  }
}

function renderAllLogsTable() {
  const tbody = document.getElementById("logs-table-body");
  if (!tbody) return;

  const query = (state.logsSearchQuery || "").toLowerCase().trim();
  const filterType = state.logsFilterType || "all";

  const filtered = state.logs.filter((log) => {
    const matchesSearch = (log.employee_id || "").toLowerCase().includes(query) ||
                          (log.name || "").toLowerCase().includes(query) ||
                          (log.department || "").toLowerCase().includes(query);
    const matchesType = filterType === "all" || log.event_type === filterType;
    return matchesSearch && matchesType;
  });

  if (!filtered.length) {
    tbody.innerHTML = '<tr><td class="empty" colspan="10">Không tìm thấy nhật ký phù hợp.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered
    .map((log) => {
      const eventLabel = log.event_type === "check_in" ? 
        '<span class="badge present">Check-in</span>' : 
        '<span class="badge pending">Check-out</span>';
      const confidencePct = Math.round((log.confidence || 0) * 100);
      const livenessPct = Math.round((log.liveness_score || 0) * 100);
      const livenessBadge = log.is_real_face ? 
        `<span class="badge present" style="background:#e0f2fe;color:#0369a1;border-color:#bae6fd;">${livenessPct}% (Live)</span>` : 
        `<span class="badge late" style="background:#fee2e2;color:#991b1b;border-color:#fca5a5;font-weight:bold;">CANH BAO GIA MAO (${livenessPct}%)</span>`;

      // Emotion badge
      const emotion = (log.emotion || 'neutral').toLowerCase();
      const emotionEmojis = {
        neutral: '&#128528;', happiness: '&#128522;', surprise: '&#128558;',
        sadness: '&#128546;', anger: '&#128544;', disgust: '&#129314;',
        contempt: '&#128528;', fear: '&#128552;'
      };
      const emotionLabels = {
        neutral: 'Binh thuong', happiness: 'Vui ve', surprise: 'Ngac nhien',
        sadness: 'Buon ba', anger: 'Tuc gian', disgust: 'Kho chiu',
        contempt: 'Kinh thuong', fear: 'So hai'
      };
      const emotionBgColors = {
        neutral: '#f1f5f9', happiness: '#dcfce7', surprise: '#e0f2fe',
        sadness: '#eff6ff', anger: '#fee2e2', disgust: '#f0fdf4',
        contempt: '#faf5ff', fear: '#fef3c7'
      };
      const emotionTextColors = {
        neutral: '#475569', happiness: '#166534', surprise: '#0369a1',
        sadness: '#1d4ed8', anger: '#991b1b', disgust: '#15803d',
        contempt: '#7e22ce', fear: '#92400e'
      };
      const emBg = emotionBgColors[emotion] || '#f1f5f9';
      const emColor = emotionTextColors[emotion] || '#475569';
      const emEmoji = emotionEmojis[emotion] || '&#128528;';
      const emLabel = emotionLabels[emotion] || emotion;
      const emotionBadge = `<span class="emotion-badge" style="background:${emBg};color:${emColor};">${emEmoji} ${emLabel}</span>`;
      
      const trClass = log.is_real_face ? "" : ' class="spoof-row"';
        
      return `
        <tr${trClass}>
          <td><strong>${log.employee_id}</strong></td>
          <td>${log.name || log.employee_id}</td>
          <td>${log.department || "-"}</td>
          <td>${eventLabel}</td>
          <td>${log.time || "-"}</td>
          <td>${log.date || "-"}</td>
          <td>${confidencePct}%</td>
          <td>${livenessBadge}</td>
          <td>${emotionBadge}</td>
          <td><code style="font-size:11px;">${log.camera_id || "-"}</code></td>
        </tr>
      `;
    })
    .join("");
}

function switchSubView(subviewId) {
  document.querySelectorAll("#employees-view .sub-view").forEach((sv) => {
    sv.style.display = sv.id === subviewId ? "block" : "none";
  });

  document.querySelectorAll(".sub-nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.subview === subviewId);
  });

  if (subviewId === "employee-register-panel" && registration.mode === "camera") {
    startCamera();
  } else {
    stopCamera();
  }
}

function switchView(viewName) {
  if (viewName !== "employees") {
    stopCamera();
  }

  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.getElementById(`${viewName}-view`).classList.add("active");

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewName);
  });

  if (viewName === "employees") {
    const activeSubBtn = document.querySelector(".sub-nav-item.active");
    const activeSubview = activeSubBtn ? activeSubBtn.dataset.subview : "employee-list-panel";
    switchSubView(activeSubview);
  } else if (viewName === "settings") {
    loadSettings();
  } else if (viewName === "logs") {
    loadAllLogs();
  }
}

function initEvents() {
  els.apiBase.value = state.apiBase;
  els.dateInput.value = state.date;

  els.saveApi.addEventListener("click", () => {
    state.apiBase = els.apiBase.value.trim() || "http://127.0.0.1:8000";
    localStorage.setItem("camai.apiBase", state.apiBase);
    loadData();
    initWebSocket();
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

  document.querySelectorAll(".sub-nav-item").forEach((button) => {
    button.addEventListener("click", () => switchSubView(button.dataset.subview));
  });

  els.modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      registration.mode = mode;
      els.modeBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (mode === "camera") {
        els.cameraMode.style.display = "block";
        els.uploadMode.style.display = "none";
        startCamera();
      } else {
        els.cameraMode.style.display = "none";
        els.uploadMode.style.display = "block";
        stopCamera();
      }
    });
  });

  els.startCameraBtn.addEventListener("click", (e) => {
    e.preventDefault();
    startCamera();
  });

  els.stopCameraBtn.addEventListener("click", (e) => {
    e.preventDefault();
    stopCamera();
  });

  els.captureBtn.addEventListener("click", (e) => {
    e.preventDefault();
    capturePhoto();
  });

  els.uploadTriggerBtn.addEventListener("click", (e) => {
    e.preventDefault();
    els.fileInput.click();
  });

  els.fileInput.addEventListener("change", (e) => {
    handleFileUpload(e.target.files);
  });

  els.registerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitRegistration();
  });

  const searchInput = document.getElementById("employee-search");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.employeeSearchQuery = e.target.value;
      renderEmployees();
    });
  }

  const settingsForm = document.getElementById("settings-form");
  if (settingsForm) {
    settingsForm.addEventListener("submit", saveSettings);
  }

  const logsSearch = document.getElementById("logs-search");
  if (logsSearch) {
    logsSearch.addEventListener("input", (e) => {
      state.logsSearchQuery = e.target.value;
      renderAllLogsTable();
    });
  }

  const logsFilterType = document.getElementById("logs-filter-type");
  if (logsFilterType) {
    logsFilterType.addEventListener("change", (e) => {
      state.logsFilterType = e.target.value;
      renderAllLogsTable();
    });
  }

  renderPhotos();
}

function startClock() {
  const tick = () => {
    els.clock.textContent = new Date().toLocaleTimeString("vi-VN");
  };
  tick();
  setInterval(tick, 1000);
}

let wsConn = null;
function initWebSocket() {
  if (wsConn) {
    try {
      wsConn.close();
    } catch(e){}
  }
  
  try {
    const apiBaseUrl = state.apiBase.trim();
    let wsUrl;
    if (apiBaseUrl.startsWith("http")) {
      const urlObj = new URL(apiBaseUrl);
      const protocol = urlObj.protocol === "https:" ? "wss:" : "ws:";
      wsUrl = `${protocol}//${urlObj.host}/ws`;
    } else {
      wsUrl = "ws://127.0.0.1:8000/ws";
    }
    
    console.log("[WS] Connecting to:", wsUrl);
    wsConn = new WebSocket(wsUrl);
    
    wsConn.onopen = () => {
      console.log("[WS] Connection established successfully!");
    };
    
    wsConn.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log("[WS] Received message:", msg);
        if (msg.type === "attendance_event") {
          console.log("[WS] Live attendance event logged:", msg.data);
          loadData();
          const logsView = document.getElementById("logs-view");
          if (logsView && logsView.classList.contains("active")) {
            loadAllLogs();
          }
        }
      } catch (e) {
        console.error("[WS] Error parsing WebSocket message:", e);
      }
    };
    
    wsConn.onerror = (err) => {
      console.warn("[WS] Connection error:", err);
    };
    
    wsConn.onclose = () => {
      console.log("[WS] Connection closed. Attempting reconnect in 5s...");
      setTimeout(initWebSocket, 5000);
    };
  } catch(error) {
    console.error("[WS] Initialization error:", error);
    setTimeout(initWebSocket, 5000);
  }
}

initEvents();
startClock();
loadData();
initWebSocket();
setInterval(loadData, 30000); // We have websockets now, so we can decrease polling frequency to 30s as fallback!
