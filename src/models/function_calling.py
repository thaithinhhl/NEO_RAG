from typing import Optional, Dict, Any
import json
from langchain_ollama import OllamaLLM
import re
import uuid
from datetime import datetime
import os
import sys
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.chat_history import ChatHistory

class FunctionCalling:
    def __init__(self, tools_config_path: str = "config/tools.json", chat_history: Optional[ChatHistory] = None):
        self.tools = json.load(open(tools_config_path, "r", encoding="utf-8"))
        self.llm = OllamaLLM(
            model="llama3.1:8b",  
            temperature=0.1,      
            top_k=10,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,
            stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:', '```']
        )
        self.chat_history = chat_history
        
    def tinh_thoi_gian_thu_viec(self, vi_tri_cong_viec: str) -> str:
        """Tính thời gian thử việc theo vị trí công việc"""
        periods = {
            "Quản lý": "60 ngày", 
            "Thực tập": "3 đến 6 tháng"
        }
        return periods.get(vi_tri_cong_viec, "20 ngày")

    def nhan_su_cong_ty_NEO(self, phong_ban: str) -> str:
        """Trả về thông tin nhân sự theo phòng ban"""
        if phong_ban == "kỹ thuật" or phong_ban == 'kĩ thuật':
            return "Có trên 30 nhân viên kỹ thuật tại công ty NEO"
        elif phong_ban == "quản lý":
            return "Có trên 10 nhân viên quản lý tại công ty NEO"
        else:
            return "Không có thông tin về nhân sự phòng ban này"

    def thoi_gian_lam_viec_tai_cong_ty_NEO(self, xem_gio_lam: bool) -> str:
        """Trả về thông tin thời gian làm việc"""
        return '''Thời gian làm việc tại NEO:
                - Sáng: 8h - 12h
                - Chiều: 13h - 17h
                - Ngày làm việc: Thứ 2 đến thứ 6
                - Thời gian nghỉ trưa: 12h - 13h'''

    def dia_chi_cong_ty_NEO(self, xem_dia_chi: bool) -> str:
        """Trả về địa chỉ công ty"""
        return """Địa chỉ công ty NEO:
                - Số 1 Phùng Chí Kiên
                - Phường Nghĩa Đô
                - Quận Cầu Giấy
                - Hà Nội"""

    def tinh_ngay_nghi_phep_con_lai_NEO(self, so_ngay_da_nghi: int) -> str:
        """Tính số ngày nghỉ phép còn lại"""
        tong_ngay_nghi = 12  
        ngay_con_lai = tong_ngay_nghi - so_ngay_da_nghi
        
        if ngay_con_lai < 0:
            return f"Bạn đã sử dụng quá số ngày nghỉ phép trong năm. Đã nghỉ: {so_ngay_da_nghi} ngày, vượt quá {abs(ngay_con_lai)} ngày"
        
        return f"""Thông tin ngày nghỉ:
                - Tổng số ngày nghỉ phép trong năm: {tong_ngay_nghi} ngày
                - Số ngày đã nghỉ: {so_ngay_da_nghi} ngày
                - Số ngày còn lại: {ngay_con_lai} ngày"""

    def execute_function(self, func_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Thực thi function theo tên và tham số"""
        function_map = {
            "tinh_thoi_gian_thu_viec": self.tinh_thoi_gian_thu_viec,
            "nhan_su_cong_ty_NEO": self.nhan_su_cong_ty_NEO,
            "dia_chi_cong_ty_NEO": self.dia_chi_cong_ty_NEO,
            "thoi_gian_lam_viec_tai_cong_ty_NEO": self.thoi_gian_lam_viec_tai_cong_ty_NEO,
            "tinh_ngay_nghi_phep_con_lai_NEO": self.tinh_ngay_nghi_phep_con_lai_NEO
        }
        
        if func_name not in function_map:
            return None
        
        try:
            result = function_map[func_name](**arguments)
            return result
        except Exception as e:
            print(f"Lỗi khi thực thi function {func_name}: {str(e)}")
            return None

    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
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
            # Tìm chuỗi nằm giữa { và } cuối cùng
            start = response.rfind('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and start < end:
                try:
                    json_str = response[start:end+1]
                    return json.loads(json_str)
                except:
                    pass
                    
            print("\nKhông tìm thấy JSON hợp lệ trong response:", response)
            return None
            
        except Exception as e:
            print(f"\nLỗi khi parse JSON: {str(e)}")
            print("Response gốc:", response)
            return None

    def get_purpose(self, func_name: str) -> str:
        """Lấy mục đích sử dụng của function"""
        for tool in self.tools:
            if tool.get("name") == func_name: 
                purpose = tool.get("user_question", "để trả lời chính xác")
                return purpose
        return "để trả lời chính xác"

    def get_user_friendly_name(self, func_name: str, param_name: str) -> str:
        """Chuyển đổi tên parameter sang user-friendly"""
        for tool in self.tools:
            if tool.get("name") == func_name:
                properties = tool.get("parameters", {}).get("properties", {})
                if param_name in properties:
                    description = properties[param_name].get("description")
                    if description:
                        return description  
                break
        return param_name

    def create_prompt(self, query: str, user_id: str = None) -> str:
        """Tạo prompt cho LLM"""
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Tools: function_calling
NHIỆM VỤ CỦA BẠN:
Phân tích câu hỏi và trả về JSON theo định dạng sau:
{{
    "function": "tên_hàm",
    "arguments": {{
        Nếu có giá trị -> điền giá trị
        Nếu không có giá trị -> điền null cho string, null cho number
    }},
    "missing_info": ["danh_sách_tham_số_thiếu"]
}}

QUY TẮC NGHIÊM NGẶT:
1. CHỈ TRẢ VỀ MỘT JSON DUY NHẤT
2. KHÔNG THÊM TEXT GIẢI THÍCH
3. KHÔNG THÊM MARKDOWN
4. KHÔNG THÊM NEWLINE
5. KHÔNG THÊM KHOẢNG TRẮNG THỪA

LOGIC XỬ LÝ:
- Nếu đủ thông tin để gọi hàm: điền "function" và "arguments", để "missing_info": []
- Nếu thiếu thông tin: điền "function", điền "arguments" với giá trị mặc định (null/0), điền "missing_info" với danh sách tham số thiếu
- Nếu không xác định được hàm hoặc câu hỏi không liên quan đến tính toán/tra cứu: trả về {{"function": "Not_call_function_calling", "arguments": {{}}, "missing_info": []}}

CÁC HÀM CÓ THỂ GỌI:
{json.dumps(self.tools, ensure_ascii=False, indent=2)}
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

    def process_query(self, query: str, user_id: str = None) -> Optional[str]:
        try:
            time_start = time.time()
            prompt = self.create_prompt(query, user_id)
            response = self.llm.invoke(prompt)
            parsed = self.extract_json_from_response(response)
            if not parsed:
                error_msg = "Lỗi: Không thể phân tích phản hồi từ LLM. Vui lòng thử lại."
                return error_msg
            func_name = parsed.get("function")
            arguments = parsed.get("arguments", {})
            missing_params = parsed.get("missing_info", [])

            if func_name == 'Not_call_function_calling':
                return None

            # enought info to call function
            if not missing_params:
                func_name_result = self.execute_function(func_name, arguments)
                if func_name_result:
                    return func_name_result
                else:
                    return "Lỗi: Không thể thực hiện hàm."
                
            # not enought info to call function
            purpose = self.get_purpose(func_name)
            # convert required -> user-friendly 
            context_friendly = []
            for param in missing_params:
                friendly_name = self.get_user_friendly_name(func_name, param)
                context_friendly.append(friendly_name)
            
            if len(context_friendly) == 1:
                response = f"Bạn có thể cho tôi biết thông tin thêm về {context_friendly[0]} {purpose} không?"
            else:
                missing_str = ", ".join(context_friendly[:-1]) + f" và {context_friendly[-1]}"
                response = f"Bạn có thể cho tôi biết thêm {missing_str} {purpose} không?"
                
            return response
            
        except Exception as e:
            return f"Lỗi: Không thể xử lý câu hỏi. Vui lòng thử lại. Chi tiết: {str(e)}"
        finally:
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
