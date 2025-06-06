from typing import Optional, Dict, Any
import json
from langchain_community.llms.ollama import Ollama
import re
import uuid
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.chat_history import message_history, get_history, delete_history

TOOLS = [
    {
        "name": "tinh_thoi_gian_thu_viec",
        "description": "Tính thời gian thử việc tối đa theo loại công việc",
        "parameters": {
            "type": "object",
            "properties": {
                "job_type": {
                    "type": "string",
                    "enum": ["ky_thuat_cao", "quan_ly", "thuc_tap"],
                    "description": "Loại công việc: ky_thuat_cao (công việc có chuyên môn kỹ thuật cao), quan_ly (quản lý), thuc_tap (thực tập)"
                }
            },
            "required": ["job_type"]
        },
        "context_requirements": ["thử việc", "thời gian thử việc"]
    },
    {
        "name": "tra_cuu_luong_toi_thieu",
        "description": "Tra cứu mức lương tối thiểu vùng hiện hành tại Việt Nam",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "enum": ["vung_I", "vung_II", "vung_III", "vung_IV"],
                    "description": "Vùng cần tra cứu: vung_I, vung_II, vung_III, vung_IV"
                }
            },
            "required": ["region"],
        },
        "context_requirements": ["lương tối thiểu", "lương cơ bản vùng"]
    },
    {
        "name": "kiem_tra_gio_lam_them",
        "description": "Kiểm tra giới hạn làm thêm giờ theo quy định của Bộ luật lao động",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["ngay", "thang", "nam"],
                    "description": "Khoảng thời gian cần kiểm tra: ngay, thang, nam"
                }
            },
            "required": ["period"]
        },
        "context_requirements": ["làm thêm giờ", "tăng ca", "overtime"]
    },
    {
        "name": "tinh_luong_thuc_nhan",
        "description": "Tính tiền lương thực nhận sau khi trừ các khoản bảo hiểm bắt buộc (BHXH, BHYT, BHTN) và thuế TNCN",
        "parameters": {
            "type": "object",
            "properties": {
                "gross_salary": {"type": "number", "description": "Lương gross (VNĐ)"},
                "num_dependents": {"type": "integer", "description": "Số người phụ thuộc", "default": 0}
            },
            "required": ["gross_salary"]
        },
        "context_requirements": ["lương thực nhận", "lương net", "lương sau thuế"]
    },
    {
        "name": "tinh_ngay_phep_nam",
        "description": "Tính số ngày nghỉ phép năm theo thâm niên làm việc (không bao gồm nghỉ việc riêng như cưới, ma chay)",
        "parameters": {
            "type": "object",
            "properties": {
                "working_years": {"type": "number", "description": "Số năm làm việc"},
                "special_condition": {"type": "boolean", "description": "Làm việc nặng nhọc, độc hại, nơi có điều kiện đặc biệt?", "default": False}
            },
            "required": ["working_years"]
        },
        "context_requirements": ["nghỉ phép năm", "phép năm", "annual leave"]
    },
    {
        "name": "tinh_luong_lam_them",
        "description": "Tính tiền lương làm thêm giờ theo quy định",
        "parameters": {
            "type": "object",
            "properties": {
                "base_salary": {"type": "number", "description": "Lương cơ bản (VNĐ)"},
                "hours": {"type": "number", "description": "Số giờ làm thêm"},
                "overtime_type": {"type": "string", "enum": ["ngay_thuong", "ngay_nghi", "ngay_le"], "description": "Loại ngày làm thêm: ngày thường, ngày nghỉ, ngày lễ"}
            },
            "required": ["base_salary", "hours", "overtime_type"]
        },
        "context_requirements": ["lương làm thêm", "tiền làm thêm giờ", "tiền tăng ca"]
    },
    {
        "name": "kiem_tra_dieu_kien_nghi_viec_hop_phap",
        "description": "Kiểm tra điều kiện nghỉ việc (chấm dứt HĐLĐ) hợp pháp theo Bộ luật lao động",
        "parameters": {
            "type": "object",
            "properties": {
                "notice_days": {"type": "integer", "description": "Số ngày báo trước"},
                "reason": {"type": "string", "description": "Lý do nghỉ việc"}
            },
            "required": ["notice_days", "reason"]
        },
        "context_requirements": ["nghỉ việc", "thôi việc", "chấm dứt hợp đồng"]
    },
    {
        "name": "tinh_luong_ngay_nghi_le_tet",
        "description": "Tính tiền lương ngày nghỉ lễ, tết theo quy định",
        "parameters": {
            "type": "object",
            "properties": {
                "base_salary": {"type": "number", "description": "Lương cơ bản (VNĐ)"},
                "days": {"type": "number", "description": "Số ngày làm việc trong dịp lễ, tết"}
            },
            "required": ["base_salary", "days"]
        },
        "context_requirements": ["lương ngày lễ", "lương ngày tết", "nghỉ lễ tết"]
    },
    {
        "name": "kiem_tra_dieu_kien_nghi_om_huong_bhxh",
        "description": "Kiểm tra điều kiện nghỉ ốm hưởng BHXH theo quy định",
        "parameters": {
            "type": "object",
            "properties": {
                "bhxh_months": {"type": "integer", "description": "Số tháng đã đóng BHXH"},
                "has_medical_certificate": {"type": "boolean", "description": "Có giấy chứng nhận nghỉ ốm hợp lệ không?"}
            },
            "required": ["bhxh_months", "has_medical_certificate"]
        },
        "context_requirements": ["nghỉ ốm", "hưởng bảo hiểm xã hội", "nghỉ bệnh"]
    }
]

def tinh_thoi_gian_thu_viec(job_type: str) -> str:
    periods = {"ky_thuat_cao": "60 ngày", "quan_ly": "60 ngày", "thuc_tap": "3 đến 6 tháng "}
    return periods.get(job_type, "30 ngày")

def tra_cuu_luong_toi_thieu(region: str) -> str:
    wages = {"vung_I": "4.680.000 đồng/tháng", "vung_II": "4.160.000 đồng/tháng",
             "vung_III": "3.640.000 đồng/tháng", "vung_IV": "3.250.000 đồng/tháng"}
    return wages.get(region, "Không tìm thấy thông tin vùng")

def kiem_tra_gio_lam_them(period: str) -> str:
    limits = {"ngay": "Không quá 12 giờ trong 1 ngày", "thang": "Không quá 40 giờ trong 1 tháng",
              "nam": "Không quá 200 giờ trong 1 năm, trường hợp đặc biệt không quá 300 giờ"}
    return limits.get(period, "Không có thông tin về giới hạn thời gian này")

def tinh_luong_thuc_nhan(gross_salary: float, num_dependents: int = 0) -> str:
    bhxh = 0.08
    bhyt = 0.015
    bhtn = 0.01
    total_insurance = bhxh + bhyt + bhtn
    insurance_amount = gross_salary * total_insurance
    salary_after_insurance = gross_salary - insurance_amount
    personal_deduction = 11000000
    dependent_deduction = 4400000 * num_dependents
    taxable_income = max(0, salary_after_insurance - personal_deduction - dependent_deduction)
    tax = taxable_income * 0.05 if taxable_income > 0 else 0
    net_salary = salary_after_insurance - tax
    return f"Lương thực nhận: {net_salary:,.0f} VNĐ (đã trừ bảo hiểm và thuế TNCN bậc 1 nếu có)"

def tinh_ngay_phep_nam(working_years: float, special_condition: bool = False) -> str:
    base_days = 14 if special_condition else 12
    extra_days = int(working_years // 5)
    total_days = base_days + extra_days
    return f"Số ngày phép năm: {total_days} ngày "

def tinh_luong_lam_them(base_salary: float, hours: float, overtime_type: str) -> str:
    if overtime_type == "ngay_thuong":
        rate = 1.5
    elif overtime_type == "ngay_nghi":
        rate = 2.0
    elif overtime_type == "ngay_le":
        rate = 3.0
    else:
        return "Loại ngày làm thêm không hợp lệ."
    pay = base_salary * rate * hours
    return f"Tiền lương làm thêm giờ: {pay:,.0f} VNĐ (hệ số {rate}x)"

def kiem_tra_dieu_kien_nghi_viec_hop_phap(notice_days: int, reason: str) -> str:
    if notice_days >= 30 or reason.lower() in ["bị ngược đãi", "không được trả lương", "bị quấy rối"]:
        return "Đủ điều kiện nghỉ việc hợp pháp theo BLLĐ."
    else:
        return "Chưa đủ điều kiện nghỉ việc hợp pháp (cần báo trước đủ số ngày hoặc có lý do chính đáng)."

def tinh_luong_ngay_nghi_le_tet(base_salary: float, days: float) -> str:
    try:
        # Validate input
        if not isinstance(base_salary, (int, float)) or base_salary < 0:
            return "Lỗi: Mức lương không hợp lệ"
        if not isinstance(days, (int, float)) or days < 0:
            return "Lỗi: Số ngày không hợp lệ"
            
        # Tính lương (300% lương cơ bản)
        pay = float(base_salary) * 3 * float(days)
        
        # Format kết quả với dấu phẩy ngăn cách hàng nghìn
        formatted_pay = "{:,.0f}".format(pay)
        formatted_base = "{:,.0f}".format(base_salary)
        
        return f"Tiền lương ngày nghỉ lễ, tết: {formatted_pay} VNĐ (300% của {formatted_base} VNĐ × {days} ngày)"
    except Exception as e:
        return f"Lỗi khi tính lương: {str(e)}"

def kiem_tra_dieu_kien_nghi_om_huong_bhxh(bhxh_months: int, has_medical_certificate: bool) -> str:
    if bhxh_months >= 6 and has_medical_certificate:
        return "Đủ điều kiện nghỉ ốm hưởng BHXH."
    else:
        return "Chưa đủ điều kiện nghỉ ốm hưởng BHXH (cần đủ số tháng đóng và giấy tờ hợp lệ)."

def execute_function(func_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    function_map = {
        "tinh_thoi_gian_thu_viec": tinh_thoi_gian_thu_viec,
        "tra_cuu_luong_toi_thieu": tra_cuu_luong_toi_thieu,
        "kiem_tra_gio_lam_them": kiem_tra_gio_lam_them,
        "tinh_luong_thuc_nhan": tinh_luong_thuc_nhan,
        "tinh_ngay_phep_nam": tinh_ngay_phep_nam,
        "tinh_luong_lam_them": tinh_luong_lam_them,
        "kiem_tra_dieu_kien_nghi_viec_hop_phap": kiem_tra_dieu_kien_nghi_viec_hop_phap,
        "tinh_luong_ngay_nghi_le_tet": tinh_luong_ngay_nghi_le_tet,
        "kiem_tra_dieu_kien_nghi_om_huong_bhxh": kiem_tra_dieu_kien_nghi_om_huong_bhxh
    }
    if func_name not in function_map:
        return None
    try:
        result = function_map[func_name](**arguments)
        return result
    except Exception as e:
        print(f"Lỗi khi thực thi function {func_name}: {str(e)}")
        return None

# ------------ xử lý phần định dạng json từ response ------------
def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Trích xuất JSON từ response của LLM, xử lý các trường hợp:
    1. Response chỉ chứa JSON
    2. Response có giải thích + JSON
    3. Response có nhiều JSON (lấy cái cuối cùng)
    """
    try:
        # Thử parse trực tiếp nếu response là JSON
        try:
            return json.loads(response)
        except:
            pass
            
        # Tìm tất cả các JSON blocks trong response
        json_blocks = re.findall(r'\{(?:[^{}]|(?R))*\}', response)
        if json_blocks:
            # Lấy JSON block cuối cùng và parse
            try:
                return json.loads(json_blocks[-1])
            except:
                pass
                
        # Tìm JSON block cuối cùng bằng regex khác
        match = re.search(r'\{[\s\S]*\}(?=[^{]*$)', response)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
                
        print("\nKhông tìm thấy JSON hợp lệ trong response:", response)
        return None
        
    except Exception as e:
        print(f"\nLỗi khi parse JSON: {str(e)}")
        print("Response gốc:", response)
        return None

def process_query(query: str, user_id: str = None) -> Optional[str]:
    try:
        llm = Ollama(
            model="llama3.1:8b",  
            temperature=0.1,      
            top_k=10,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,
            stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:', '```']
        )        
        if not user_id:
            user_id = str(uuid.uuid4())
            
        history = get_history(user_id)
        prompt = f'''Bạn là một luật sư chuyên nghiệp tại Việt Nam, hỗ trợ tính toán, phân tích và tra cứu quy định lao động.

NHIỆM VỤ CỦA BẠN:
Phân tích câu hỏi và trả về chính xác JSON theo định dạng sau, không thêm bất kỳ text nào khác:

{{
    "function": "<tên_hàm>",
    "arguments": {{<tham_số_hàm>}},
    "missing_info": ["<danh_sách_tham_số_thiếu>"]
}}

QUY TẮC NGHIÊM NGẶT:
1. CHỈ TRẢ VỀ MỘT JSON DUY NHẤT
2. KHÔNG THÊM TEXT GIẢI THÍCH
3. KHÔNG THÊM MARKDOWN
4. KHÔNG THÊM NEWLINE
5. KHÔNG THÊM KHOẢNG TRẮNG THỪA

LOGIC XỬ LÝ:
- Nếu đủ thông tin để gọi hàm: điền "function" và "arguments", để "missing_info": []
- Nếu thiếu thông tin: điền "function", để "arguments": {{}}, điền "missing_info" với danh sách tham số thiếu
- Nếu không xác định được hàm hoặc câu hỏi không liên quan đến tính toán/tra cứu: trả về {{"function": "Not_call_function_calling", "arguments": {{}}, "missing_info": []}}

ÁNH XẠ TỪ KHÓA:
- 'kỹ thuật cao', 'kỹ sư' -> 'ky_thuat_cao'
- 'vùng 1', 'vùng miền bắc' -> 'vung_I'
- 'ngày chủ nhật','cuối tuần' -> 'ngay_nghi'
- 'Tết', 'Quốc Khánh' -> 'ngay_le'

CÁC HÀM CÓ THỂ GỌI:
{json.dumps(TOOLS, ensure_ascii=False, indent=2)}

Câu hỏi: {query}

Lịch sử trao đổi:
{history}'''

        print("DEBUG LLM PROMPT:", prompt)
        response = llm.invoke(prompt)
        print("DEBUG LLM RESPONSE:", response)

        result = extract_json_from_response(response)
        if not result:
            error_msg = "Lỗi: Không thể phân tích phản hồi từ LLM. Vui lòng thử lại."
            return error_msg

        func_name = result.get("function")
        arguments = result.get("arguments", {})
        missing_params = result.get("missing_info", []) 

        if func_name == 'Not_call_function_calling':
            return None

# ------------- Đủ thông tin để gọi hàm ---------------
        if not missing_params:
            func_name_result = execute_function(func_name, arguments)
            if func_name_result:
                return func_name_result
            else:
                return "Lỗi: Không thể thực hiện hàm."
            
# ------------- Không đủ thông tin để gọi hàm ---------------
        if missing_params:
            param_descriptions = {
                'job_type': 'vị trí công việc',
                "base_salary": "mức lương cơ bản",
                "hours": "số giờ làm thêm",
                "overtime_type": "loại ngày làm thêm (ngày thường/ngày nghỉ/ngày lễ)",
                "working_years": "số năm làm việc",
                "average_salary": "mức lương bình quân",
                "num_dependents": "số người phụ thuộc",
                "gross_salary": "mức lương tổng (gross)",
                "days": "số ngày",
                "notice_days": "số ngày báo trước",
                "reason": "lý do",
                "contract_count": "số lần đã ký hợp đồng xác định thời hạn",
                "job_type": "loại công việc",
                "region": "vùng/khu vực",
                "period": "khoảng thời gian",
                "special_condition": "điều kiện đặc biệt",
                "has_medical_certificate": "có giấy chứng nhận y tế",
                "bhxh_months": "số tháng đóng BHXH"
            }

            missing_info = [param_descriptions.get(param, param) for param in missing_params]
            
            function_questions = {
                'tinh thoi gian thu viec': 'để tính thời gian thử việc',
                "tinh_luong_thuc_nhan": "để tính lương thực nhận",
                "tinh_tro_cap_thoi_viec": "để tính trợ cấp thôi việc",
                "tinh_ngay_phep_nam": "để tính số ngày phép năm",
                "kiem_tra_dieu_kien_ky_hop_dong_khong_xac_dinh_thoi_han": "để kiểm tra điều kiện ký HĐLĐ không xác định thời hạn",
                "tinh_luong_lam_them": "để tính lương làm thêm giờ",
                "tinh_tro_cap_thai_san": "để tính trợ cấp thai sản",
                "kiem_tra_dieu_kien_nghi_viec_hop_phap": "để kiểm tra điều kiện nghỉ việc",
                "tinh_luong_ngay_nghi_le_tet": "để tính lương ngày lễ tết",
                "kiem_tra_dieu_kien_nghi_om_huong_bhxh": "để kiểm tra điều kiện nghỉ ốm hưởng BHXH",
                "tinh_thoi_gian_thu_viec": "để tính thời gian thử việc",
                "tra_cuu_luong_toi_thieu": "để tra cứu lương tối thiểu",
                "kiem_tra_gio_lam_them": "để kiểm tra giới hạn giờ làm thêm"
            }
            
            purpose = function_questions.get(func_name, "để trả lời chính xác")
            
            if len(missing_info) == 1:
                response = f"Bạn có thể cho tôi biết {missing_info[0]} {purpose} không?"
            else:
                missing_str = ", ".join(missing_info[:-1]) + f" và {missing_info[-1]}"
                response = f"Bạn có thể cho tôi biết thêm {missing_str} {purpose} không?"
            
            return response
        
    except Exception as e:
        return f"Lỗi: Không thể xử lý câu hỏi. Vui lòng thử lại. Chi tiết: {str(e)}"
