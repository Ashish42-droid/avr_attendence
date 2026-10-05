import os
import time
from datetime import datetime, timedelta
import csv
from typing import Dict, Any, Optional

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


class AttendanceManager:
    """
    Manages daily attendance recording in Excel (.xlsx) format with:
      - Automatic daily file rollover (attendance/Attendance_YYYY-MM-DD.xlsx).
      - In-memory caching & preloading of existing entries.
      - Smart cooldown and dual-punch logic (In-Time and Out-Time).
      - Lock-safe writing (graceful fallback to CSV if Excel is open).
      - Beautiful header styling and auto-sized columns.
    """

    def __init__(
        self,
        attendance_dir: str = "attendance",
        cooldown_seconds: int = 60,
        out_time_threshold_seconds: int = 120,
    ):
        self.attendance_dir = attendance_dir
        self.cooldown_seconds = cooldown_seconds
        self.out_time_threshold_seconds = out_time_threshold_seconds
        
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # State: current date string and in-memory cache of punches today
        # Map: name -> {"row_idx": int, "in_time": datetime, "out_time": datetime, "last_scan": datetime}
        self.current_date_str = ""
        self.daily_records: Dict[str, Dict[str, Any]] = {}
        
        # UI Banner state for camera overlay
        self.active_banner: Optional[Dict[str, Any]] = None
        self.banner_display_duration = 3.5  # seconds
        
        self._check_date_rollover()

    def _get_today_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _get_excel_path(self, date_str: str) -> str:
        return os.path.join(self.attendance_dir, f"Attendance_{date_str}.xlsx")

    def _get_csv_backup_path(self, date_str: str) -> str:
        return os.path.join(self.attendance_dir, f"Attendance_{date_str}_backup.csv")

    def _check_date_rollover(self):
        """Checks if date has changed and initializes/loads today's file."""
        today = self._get_today_str()
        if today != self.current_date_str:
            self.current_date_str = today
            self.daily_records.clear()
            self._init_daily_file(today)
            self._load_existing_records(today)

    def _init_daily_file(self, date_str: str):
        """Creates the Excel file for the given date with professional headers if not exists."""
        if not OPENPYXL_AVAILABLE:
            return

        excel_path = self._get_excel_path(date_str)
        if not os.path.exists(excel_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = f"Attendance {date_str}"
            
            # Title banner
            ws.merge_cells("A1:G1")
            title_cell = ws["A1"]
            title_cell.value = f"DAILY ATTENDANCE RECORD - {date_str}"
            title_cell.font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
            title_cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            title_cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 35

            # Column Headers
            headers = ["S.No", "Date", "Name", "In-Time", "Out-Time", "Duration", "Status"]
            ws.append([])  # Row 2 blank separator
            ws.append(headers)  # Row 3 headers
            ws.row_dimensions[3].height = 24

            header_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
            header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            thin_border = Border(
                left=Side(style='thin', color='BFBFBF'),
                right=Side(style='thin', color='BFBFBF'),
                top=Side(style='thin', color='BFBFBF'),
                bottom=Side(style='thin', color='BFBFBF')
            )

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=3, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

            self._auto_fit_columns(ws)
            try:
                wb.save(excel_path)
                print(f"[AttendanceManager] Initialized today's Excel sheet: {excel_path}")
            except Exception as e:
                print(f"[AttendanceManager] Error initializing {excel_path}: {e}")

    def _auto_fit_columns(self, ws):
        """Auto-adjusts column widths to fit content nicely."""
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Skip merged title row in width calculation
                if cell.row == 1:
                    continue
                if cell.value:
                    val_str = str(cell.value)
                    if len(val_str) > max_len:
                        max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    def _load_existing_records(self, date_str: str):
        """Loads records from an existing Excel sheet to avoid duplicate entries on restart."""
        if not OPENPYXL_AVAILABLE:
            return

        excel_path = self._get_excel_path(date_str)
        if not os.path.exists(excel_path):
            return

        try:
            wb = openpyxl.load_workbook(excel_path, data_only=True)
            ws = wb.active
            # Header is row 3, data starts at row 4
            for row_num in range(4, ws.max_row + 1):
                name_val = ws.cell(row=row_num, column=3).value
                in_time_val = ws.cell(row=row_num, column=4).value
                out_time_val = ws.cell(row=row_num, column=5).value

                if name_val:
                    name = str(name_val).strip()
                    in_dt = None
                    out_dt = None
                    if in_time_val and in_time_val != "-":
                        try:
                            in_dt = datetime.strptime(f"{date_str} {in_time_val}", "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass
                    if out_time_val and out_time_val != "-":
                        try:
                            out_dt = datetime.strptime(f"{date_str} {out_time_val}", "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass

                    self.daily_records[name] = {
                        "row_num": row_num,
                        "in_time": in_dt or datetime.now(),
                        "out_time": out_dt,
                        "last_scan": out_dt or in_dt or datetime.now(),
                    }
            print(f"[AttendanceManager] Loaded {len(self.daily_records)} existing attendance records for {date_str}.")
        except Exception as e:
            print(f"[AttendanceManager] Could not read existing records: {e}")

    def mark_attendance(self, name: str) -> Dict[str, Any]:
        """
        Main interface called when a face is recognized.
        Handles de-duplication, In-Time punch, Out-Time update, and Excel save.
        """
        if not name or name.strip().lower() in ["unknown", ""]:
            return {"status": "IGNORED", "message": "Unknown face"}

        clean_name = name.strip()
        self._check_date_rollover()
        now = datetime.now()
        date_str = self.current_date_str
        time_str = now.strftime("%H:%M:%S")

        # Case 1: First scan of the day for this person (IN-TIME)
        if clean_name not in self.daily_records:
            return self._record_in_time(clean_name, now, date_str, time_str)

        # Person already has a record today
        rec = self.daily_records[clean_name]
        time_since_last_scan = (now - rec["last_scan"]).total_seconds()
        time_since_in = (now - rec["in_time"]).total_seconds()

        # Case 2: Cooldown - Scanned too soon after last punch
        if time_since_last_scan < self.cooldown_seconds:
            res = {
                "status": "COOLDOWN",
                "name": clean_name,
                "in_time": rec["in_time"].strftime("%H:%M:%S"),
                "out_time": rec["out_time"].strftime("%H:%M:%S") if rec["out_time"] else "-",
                "message": f"{clean_name}: Already marked today ({rec['in_time'].strftime('%H:%M:%S')})",
                "color": (0, 200, 255) # Yellow/Amber
            }
            if not self.active_banner:
                self._set_banner(res)
            return res

        # Case 3: OUT-TIME Punch
        # Only record Out-Time if minimum threshold has passed since In-Time
        if time_since_in >= self.out_time_threshold_seconds:
            return self._record_out_time(clean_name, now, date_str, time_str, rec)
        else:
            # Within early arrival window, update last scan time without changing out-time
            rec["last_scan"] = now
            res = {
                "status": "ALREADY_PRESENT",
                "name": clean_name,
                "in_time": rec["in_time"].strftime("%H:%M:%S"),
                "message": f"{clean_name}: Already logged In at {rec['in_time'].strftime('%H:%M:%S')}",
                "color": (0, 220, 220)
            }
            if not self.active_banner:
                self._set_banner(res)
            return res

    def _record_in_time(self, name: str, now: datetime, date_str: str, time_str: str) -> Dict[str, Any]:
        """Records initial In-Time punch to Excel and memory."""
        excel_path = self._get_excel_path(date_str)
        s_no = len(self.daily_records) + 1
        row_num = s_no + 3  # row 1: title, row 2: blank, row 3: header

        record = {
            "row_num": row_num,
            "in_time": now,
            "out_time": None,
            "last_scan": now
        }
        self.daily_records[name] = record

        # Save to Excel
        saved_excel = False
        if OPENPYXL_AVAILABLE:
            try:
                if not os.path.exists(excel_path):
                    self._init_daily_file(date_str)

                wb = openpyxl.load_workbook(excel_path)
                ws = wb.active
                
                row_data = [s_no, date_str, name, time_str, "-", "-", "Present"]
                ws.append(row_data)

                # Format new row
                target_row = ws.max_row
                record["row_num"] = target_row

                font = Font(name="Calibri", size=10)
                thin_border = Border(
                    left=Side(style='thin', color='E0E0E0'),
                    right=Side(style='thin', color='E0E0E0'),
                    top=Side(style='thin', color='E0E0E0'),
                    bottom=Side(style='thin', color='E0E0E0')
                )
                
                for col_idx in range(1, len(row_data) + 1):
                    c = ws.cell(row=target_row, column=col_idx)
                    c.font = font
                    c.border = thin_border
                    c.alignment = Alignment(horizontal="center" if col_idx != 3 else "left", vertical="center")

                self._auto_fit_columns(ws)
                wb.save(excel_path)
                saved_excel = True
            except PermissionError:
                print(f"[!] [Excel Locked] '{excel_path}' is open in Excel! Saving to CSV backup...")
                self._append_to_csv_backup(date_str, [s_no, date_str, name, time_str, "-", "-", "Present"])
            except Exception as e:
                print(f"[!] Error saving to Excel: {e}")
                self._append_to_csv_backup(date_str, [s_no, date_str, name, time_str, "-", "-", "Present"])
        else:
            self._append_to_csv_backup(date_str, [s_no, date_str, name, time_str, "-", "-", "Present"])

        result = {
            "status": "MARKED_IN",
            "name": name,
            "in_time": time_str,
            "out_time": "-",
            "message": f"Welcome {name}! Attendance Marked at {time_str}",
            "color": (0, 255, 0) # Green
        }
        self._set_banner(result)
        print(f"[Attendance] [+] IN-TIME: {name} at {time_str}")
        return result

    def _record_out_time(self, name: str, now: datetime, date_str: str, time_str: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Updates Out-Time and Duration in Excel and memory."""
        rec["out_time"] = now
        rec["last_scan"] = now

        in_time_dt = rec["in_time"]
        duration = now - in_time_dt
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}h {minutes:02d}m"

        excel_path = self._get_excel_path(date_str)
        if OPENPYXL_AVAILABLE and os.path.exists(excel_path):
            try:
                wb = openpyxl.load_workbook(excel_path)
                ws = wb.active
                target_row = rec.get("row_num")
                
                # Verify row contains the correct name
                if target_row and ws.cell(row=target_row, column=3).value == name:
                    ws.cell(row=target_row, column=5, value=time_str) # Out-Time
                    ws.cell(row=target_row, column=6, value=duration_str) # Duration
                else:
                    # Search for row with matching name
                    for r in range(4, ws.max_row + 1):
                        if ws.cell(row=r, column=3).value == name:
                            ws.cell(row=r, column=5, value=time_str)
                            ws.cell(row=r, column=6, value=duration_str)
                            rec["row_num"] = r
                            break

                self._auto_fit_columns(ws)
                wb.save(excel_path)
            except PermissionError:
                print(f"[!] [Excel Locked] '{excel_path}' is open in Excel! Out-time updated in CSV backup.")
                self._append_to_csv_backup(date_str, ["UPDATE", date_str, name, rec['in_time'].strftime("%H:%M:%S"), time_str, duration_str, "Present"])
            except Exception as e:
                print(f"[!] Error updating Out-Time in Excel: {e}")

        result = {
            "status": "MARKED_OUT",
            "name": name,
            "in_time": rec["in_time"].strftime("%H:%M:%S"),
            "out_time": time_str,
            "duration": duration_str,
            "message": f"Goodbye {name}! Out-Time: {time_str} (Duration: {duration_str})",
            "color": (255, 140, 0) # Cyan / Light Blue
        }
        self._set_banner(result)
        print(f"[Attendance] [^] OUT-TIME: {name} at {time_str} (Worked: {duration_str})")
        return result

    def _append_to_csv_backup(self, date_str: str, row: list):
        """Emergency fallback logger if Excel file is locked by user."""
        csv_path = self._get_csv_backup_path(date_str)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["S.No", "Date", "Name", "In-Time", "Out-Time", "Duration", "Status"])
            writer.writerow(row)

    def _set_banner(self, punch_result: Dict[str, Any]):
        """Sets the active on-screen notification banner."""
        self.active_banner = {
            "text": punch_result["message"],
            "color": punch_result["color"],
            "expiry": time.time() + self.banner_display_duration
        }

    def draw_banner_overlay(self, frame):
        """
        Draws an elegant status banner across the top or bottom of the OpenCV video frame.
        Call this every frame inside cv2 display loop.
        """
        if not self.active_banner:
            return frame

        if time.time() > self.active_banner["expiry"]:
            self.active_banner = None
            return frame

        import cv2
        h, w = frame.shape[:2]
        banner_h = 50
        
        # Semi-transparent dark banner bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Highlight colored accent line
        accent_color = self.active_banner["color"]
        cv2.line(frame, (0, banner_h - 2), (w, banner_h - 2), accent_color, 3)

        # Text label
        msg = self.active_banner["text"]
        cv2.putText(
            frame,
            msg,
            (15, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return frame
