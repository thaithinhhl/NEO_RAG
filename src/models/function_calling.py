from typing import Optional, Dict, Any
import json
from langchain_ollama import OllamaLLM
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class FunctionCalling:
    def __init__(self, tools_config_path: str = "config/tools.json"):
        self.tools = json.load(open(tools_config_path, "r", encoding="utf-8"))
        self.llm = OllamaLLM(
            model="qwen3:0.6b",
            format="json",
            temperature=0.1,      
            top_k=5,             
            top_p=0.8,           
            repeat_penalty=1.05,  
            num_ctx=2048,        
            stop=['Question:', 'Câu hỏi:', 'Human:', 'Assistant:', '```'],
        )   
        
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

    def thoi_gian_lam_viec_tai_cong_ty_NEO(self, xem_gio_lam: bool = True) -> str:
        """Trả về thông tin thời gian làm việc"""
        return '''Thời gian làm việc tại NEO:
                - Sáng: 8h - 12h
                - Chiều: 13h - 17h
                - Ngày làm việc: Thứ 2 đến thứ 6
                - Thời gian nghỉ trưa: 12h - 13h'''

    def dia_chi_cong_ty_NEO(self, xem_dia_chi: bool = True) -> str:
        """Trả về địa chỉ công ty"""
        return """Địa chỉ công ty NEO:
                - Số 1 Phùng Chí Kiên
                - Phường Nghĩa Đô
                - Quận Cầu Giấy
                - Hà Nội"""

    def tinh_ngay_nghi_phep_con_lai_NEO(self, so_ngay_da_nghi: int) -> str:
        """Tính số ngày nghỉ phép còn lại"""
        tong_ngay_nghi = 12  
        ngay_con_lai = tong_ngay_nghi - int(so_ngay_da_nghi)
        
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
            "tinh_ngay_nghi_phep_con_lai_NEO": self.tinh_ngay_nghi_phep_con_lai_NEO,
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
        try:
            try:
                return json.loads(response)
            except:
                pass
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
            print(f"Lỗi khi trích xuất JSON: {str(e)}")
            return None

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

    def create_prompt(self, query: str) -> str:
        tools_json_str = json.dumps(self.tools, ensure_ascii=False)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là trợ lý phân tích câu hỏi để gọi hàm. Trả về JSON duy nhất theo quy tắc:

QUY TẮC:
- Chỉ trả về JSON, không thêm gì khác
- Đủ thông tin: điền "function" và "arguments", để "missing_info": []
- Thiếu thông tin: điền "function", "arguments" với null cho tham số thiếu, điền "missing_info"
- Không liên quan: {{"function": "Not_call_function_calling", "arguments": {{}}, "missing_info": []}}

HÀM CÓ SẴN: {tools_json_str}

VÍ DỤ:
User: "thời gian làm việc của công ty là gì"
Assistant: {{"function": "thoi_gian_lam_viec_tai_cong_ty_NEO", "arguments": {{"xem_gio_lam": true}}, "missing_info": []}}

User: "cho tôi hỏi về nhân sự"  
Assistant: {{"function": "nhan_su_cong_ty_NEO", "arguments": {{"phong_ban": null}}, "missing_info": ["phong_ban"]}}
<|eot_id|><|start_header_id|>user<|end_header_id|>

            {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    def process_query(self, query: str) -> Optional[str]:
        try:    
            time_start = time.time()
            prompt = self.create_prompt(query)
            response = self.llm.invoke(prompt)
            parsed = self.extract_json_from_response(response)
            print(f"parsed: {parsed}")
            if not parsed:
                error_msg = "Lỗi: Không thể phân tích phản hồi từ LLM. Vui lòng thử lại."
                return error_msg
            func_name = parsed.get("function")
            arguments = parsed.get("arguments", {})
            missing_params = parsed.get("missing_info", [])
            if func_name == 'Not_call_function_calling':
                return None

            # check if all arguments are not None
            for param_name, param_value in arguments.items():
                if param_value is None and param_name not in missing_params:
                    missing_params.append(param_name)
            print(f"missing_params: {missing_params}")
            # enought info to call function
            if not missing_params:
                func_name_result = self.execute_function(func_name, arguments)
                if func_name_result:
                    return func_name_result
                else:
                    return "Lỗi: Không thể thực hiện hàm."
                
            # not enought info to call function
            # convert required -> user-friendly 
            context_friendly = []
            for param in missing_params:
                friendly_name = self.get_user_friendly_name(func_name, param)
                context_friendly.append(friendly_name)
            
            if len(context_friendly) == 1:
                response = f"Tôi cần cung cấp thông tin thêm về {context_friendly[0]}. Bạn có thể cho tôi biết được không."
            else:
                missing_str = ", ".join(context_friendly[:-1]) + f" và {context_friendly[-1]}"
                response = f"Bạn có thể cho tôi biết thêm {missing_str} được không?"
                
            return response
            
        except Exception as e:
            return f"Lỗi: Không thể xử lý câu hỏi. Vui lòng thử lại. Chi tiết: {str(e)}"
        finally:
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
