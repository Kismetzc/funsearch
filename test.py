import logging
import traceback
import re
import ast
import textwrap
import numpy as np # 确保导入 numpy

class AdvancedPromptLLM(BaseLLM):
    """
    LLM 使用结构化提示，并带有代码验证和修复循环。
    """
    def __init__(self, samples_per_prompt: int, api_key: str, base_url: str, **kwargs):
        """ 初始化 AdvancedPromptLLM """
        kwargs['base_temperature'] = kwargs.get('base_temperature', 0.7)
        super().__init__(samples_per_prompt, api_key, base_url, **kwargs)
        logging.info("AdvancedPromptLLM initialized with Structured Prompt (V5) and auto-fixing.")

        # --- 提示词 (V5 - 结构化模板) ---
        self.additional_prompt = """
Write a Python function `priority(item: float, bins)` for online bin packing that OUTPERFORMS Best Fit algorithm.

Your goal is to create a breakthrough algorithm that reduces the average number of bins needed compared to Best Fit.

**KEY INSIGHTS:**
- Best Fit simply minimizes remaining space by scoring bins as -1*(bins-item)
- To BEAT Best Fit, you need more sophisticated bin selection logic
- Small improvements (even 0.5-1% fewer bins) would be significant

**APPROACHES TO EXPLORE:**
1. Statistical analysis of bin distribution patterns 
2. Smart weighting based on item-to-capacity ratios
3. Penalizing creation of unusable small spaces
4. Considering the relative position of bins in the distribution
5. Using polynomial or other non-linear scoring functions

**OPTIMIZATION NOTE:** 
Your code will run on an AMD EPYC processor with 8 cores.
Use vectorized numpy operations - avoid loops when possible.

**MANDATORY STRUCTURE:**
```python
def priority(item: float, bins):
    try:
        # YOUR ALGORITHM TO OUTPERFORM BEST FIT
        # Example improvement directions:
        # - Add statistical analysis of bin distributions
        # - Consider item size relative to mean/median capacity
        # - Use non-linear scoring functions
        # - Incorporate bin fullness thresholds

        # MUST RETURN A NUMPY ARRAY with same shape as bins
        return scores  # Higher score = better bin choice
    except Exception as e:
        # Fallback for ANY error
        return np.full_like(bins, -1e9) if isinstance(bins, np.ndarray) else np.array([], dtype=float)
```
COPY this structure exactly and fill in your algorithm to BEAT Best Fit.

**INSPIRATION:**
- Current Best Fit scores bins as: scores = -1*(bins - item)
- YOUR SOLUTION MUST BE DIFFERENT AND BETTER
- Consider factors beyond just remaining space after placement

"""

    def _draw_sample(self, context_prompt: str) -> str:
        """生成并验证/修复代码，尝试多次直到获得有效代码。"""
        full_prompt = f"{context_prompt}\n{self.additional_prompt}"

        # 尝试最多3次生成有效代码
        for attempt in range(3):
            logging.info(f"Advanced LLM: Generating code attempt {attempt+1}/3")
            raw_output = self._call_api(full_prompt)
            
            if not raw_output:
                logging.warning(f"Advanced LLM: Empty API response in attempt {attempt+1}")
                continue
            
            # 清理代码并提取函数体
            cleaned_output = raw_output.strip()
            # 提取markdown代码块
            match = re.search(r'```(?:python)?\s*(.*?)\s*```', cleaned_output, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_code_content = match.group(1).strip()
                if extracted_code_content.startswith('def priority'):
                    cleaned_output = extracted_code_content
                    logging.debug("Advanced LLM: Found and extracted content within code fences.")
            
            # 尝试通过AST提取函数体
            extracted_body = self._extract_code_body_via_ast(cleaned_output)
            
            if not extracted_body:
                logging.warning(f"Advanced LLM: AST extraction failed in attempt {attempt+1}")
                continue
            
            # 验证代码是否有效
            valid, issues = self._validate_priority_code(extracted_body)
            
            if valid:
                logging.info(f"Advanced LLM: Generated valid code on attempt {attempt+1}")
                return extracted_body
            
            # 如果无效，尝试修复代码
            logging.info(f"Advanced LLM: Code validation found issues: {issues}, attempting to fix")
            fixed_body = self._fix_priority_code(extracted_body)
            
            if fixed_body:
                valid_fixed, fixed_issues = self._validate_priority_code(fixed_body)
                if valid_fixed:
                    logging.info(f"Advanced LLM: Auto-fixed code in attempt {attempt+1}")
                    return fixed_body
                else:
                    logging.warning(f"Advanced LLM: Auto-fix failed, issues remain: {fixed_issues}")
            
            # 如果修复失败，添加具体错误反馈到下一轮提示
            if attempt < 2:  # 只在非最后尝试时添加
                error_feedback = f"\nYour previous code had these issues: {issues}. Please fix them and ensure your function has the required structure with proper error handling."
                full_prompt += error_feedback
        
        # 最终失败时返回安全的回退代码
        logging.error("Advanced LLM: Failed to generate valid code after 3 attempts")
        return self._get_safe_fallback_code()

        """验证priority函数代码是否满足基本要求。"""
        issues = []
        
        # 检查是否有实质性内容
        code_lines = code_body.strip().splitlines()
        non_comment_lines = [l.strip() for l in code_lines 
                           if l.strip() and not l.strip().startswith('#')]
        if len(non_comment_lines) < 2:
            issues.append("too few lines of actual code")
        
        # 检查是否有return语句
        if "return" not in code_body:
            issues.append("missing return statement")
        
        # 检查是否包含错误处理
        if "try:" not in code_body:
            issues.append("missing try-except error handling")
        
        # 检查基本语法 - 使用ast尝试解析
        try:
            ast.parse(f"def dummy(item, bins):\n{code_body}")
        except SyntaxError as e:
            issues.append(f"syntax error at line {e.lineno}: {e.msg}")
        
        # 如果代码中有'None'返回，标记问题
        if re.search(r'return\s+None', code_body):
            issues.append("explicitly returns None")
        
        # 检查是否有numpy导入相关语句
        if "np." in code_body and "import numpy" not in code_body and "import numpy as np" not in code_body:
            # 这不是错误，因为import可能在上下文中，但值得注意
            pass
        
        return len(issues) == 0, issues

        # 1. 确保基本缩进和清理
        code_body = textwrap.dedent(code_body).strip()
        
        # 2. 添加错误处理包装
        if "try:" not in code_body:
            # 缩进代码并添加try
            indented_code = textwrap.indent(code_body, '    ')
            code_body = f"try:\n{indented_code}"
            
            # 只有当代码中没有except时才添加，避免嵌套try-except
            if "except" not in code_body:
                code_body += "\nexcept Exception as e:\n    # Auto-added error handling\n    return np.full_like(bins, -1e9) if isinstance(bins, np.ndarray) else np.array([], dtype=float)"
        
        # 3. 确保有return语句
        if "return" not in code_body:
            # 尝试找到一个数组变量作为返回值
            array_vars = re.findall(r'(\w+)\s*=\s*np\.(?:array|zeros|ones|full)', code_body)
            scores_vars = re.findall(r'(\w+)\s*=\s*scores', code_body) + re.findall(r'(\w+)\s*=\s*-', code_body)
            
            # 确定返回变量
            return_var = None
            if array_vars:
                return_var = array_vars[-1]  # 使用最后创建的数组
            elif scores_vars:
                return_var = scores_vars[-1]  # 使用scores相关变量
            else:
                return_var = "-bins"  # 退化到返回-bins (Best Fit策略)
            
            # 添加到正确位置 - 在try块内的最后或在except前
            if "except" in code_body:
                # 在第一个except之前插入
                parts = code_body.split("except", 1)
                return_stmt = f"\n    # Auto-added return\n    return {return_var}\n"
                code_body = parts[0] + return_stmt + "except" + parts[1]
            else:
                # 在末尾添加
                code_body += f"\n    # Auto-added return\n    return {return_var}"
        
        # 4. 修复显式返回None的情况
        code_body = re.sub(r'return\s+None', 'return np.full_like(bins, -1e9)', code_body)
        
        # 5. 确保有异常处理中的返回语句
        if "except" in code_body:
            except_clause = code_body.split("except", 1)[1]
            if "return" not in except_clause:
                # 在except子句末尾添加返回语句
                code_body += "\n    return np.full_like(bins, -1e9) if isinstance(bins, np.ndarray) else np.array([], dtype=float)"
        
        # 确保最终缩进正确
        return textwrap.indent(code_body, "    ") if not code_body.startswith("    ") else code_body
    def _validate_priority_code(self, code_body: str) -> tuple[bool, list]:
        """验证priority函数代码是否满足基本要求和评估其是否可能超越Best Fit。"""
        issues = []
        
        # 检查是否有实质性内容
        code_lines = code_body.strip().splitlines()
        non_comment_lines = [l.strip() for l in code_lines 
                        if l.strip() and not l.strip().startswith('#')]
        if len(non_comment_lines) < 3:  # 增加到3行最小需求，确保有足够的复杂度
            issues.append("too few lines of actual code (needs at least 3 non-comment lines)")
        
        # 检查是否有return语句
        if "return" not in code_body:
            issues.append("missing return statement")
        
        # 检查是否包含错误处理
        if "try:" not in code_body:
            issues.append("missing try-except error handling")
        
        # 检查基本语法 - 使用ast尝试解析
        try:
            ast.parse(f"def dummy(item, bins):\n{code_body}")
        except SyntaxError as e:
            issues.append(f"syntax error at line {e.lineno}: {e.msg}")
        
        # 如果代码中有'None'返回，标记问题
        if re.search(r'return\s+None', code_body):
            issues.append("explicitly returns None")
        
        # 检查是否只是简单复制了Best Fit - 寻找更复杂的解决方案
        is_simple_best_fit = False
        if re.search(r'scores\s*=\s*-\s*\(\s*bins\s*-\s*item\s*\)', code_body) or \
        re.search(r'scores\s*=\s*-bins\s*\+\s*item', code_body) or \
        re.search(r'return\s*-\s*\(\s*bins\s*-\s*item\s*\)', code_body) or \
        re.search(r'return\s*-bins\s*\+\s*item', code_body):
            is_simple_best_fit = True
            issues.append("solution is equivalent to standard Best Fit algorithm")
        
        # 检查是否包含更复杂的逻辑 - 查找高级特性
        has_advanced_features = False
        
        # 检查是否使用统计或分布特性
        if re.search(r'(mean|median|std|var|percentile|quantile|average)', code_body) or \
        re.search(r'\.sort\(', code_body) or re.search(r'sorted\(', code_body):
            has_advanced_features = True
        
        # 检查是否有条件逻辑
        if code_body.count('if ') > 1:  # 允许至少1个条件判断
            has_advanced_features = True
        
        # 检查是否有阈值或非线性转换
        if re.search(r'(np\.exp|np\.log|np\.power|np\.sqrt|\*\*|threshold|factor)', code_body):
            has_advanced_features = True
        
        # 检查是否考虑填充率/利用率
        if re.search(r'(capacity|utilization|fullness|ratio|percentage)', code_body):
            has_advanced_features = True
        
        # 如果是简单Best Fit并且没有高级特性，添加警告
        if is_simple_best_fit and not has_advanced_features:
            issues.append("code needs more sophisticated logic to potentially outperform Best Fit")
        
        # 检查是否有numpy向量化操作 (适合EPYC处理器)
        if len(re.findall(r'for\s+\w+\s+in', code_body)) > 1:  # 超过1个for循环
            issues.append("relies heavily on loops instead of numpy vectorized operations")
        
        # 检查是否有numpy导入相关语句
        if "np." in code_body and "import numpy" not in code_body and "import numpy as np" not in code_body:
            # 这不是错误，因为import可能在上下文中
            pass
        
        return len(issues) == 0, issues

    def _fix_priority_code(self, code_body: str) -> str:
        """改进：修复priority函数代码，确保包含try-except结构和return语句。"""
        logging.info("Advanced LLM: Attempting to fix code with issues")
        
        # 获取原始代码的缩进级别
        lines = code_body.splitlines()
        indent = ""
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                indent = re.match(r'^(\s*)', line).group(1)
                break
        
        # 标准化缩进为4个空格
        code_body = textwrap.dedent(code_body)
        code_body = "\n".join(["    " + line.strip() for line in code_body.splitlines() if line.strip()])
        
        # 检查是否已经有try-except结构
        has_try = "try:" in code_body
        has_except = "except" in code_body
        has_return = "return" in code_body
        
        # 修复代码
        fixed_code = ""
        
        # 1. 如果没有try-except结构，添加它
        if not has_try or not has_except:
            # 完全重新构造try-except结构
            fixed_code = "    try:\n"
            
            # 添加非注释代码行，保持原有逻辑
            for line in code_body.splitlines():
                if line.strip() and not line.strip().startswith('try:') and not line.strip().startswith('except'):
                    fixed_code += "        " + line.strip() + "\n"
            
            # 如果没有return语句，添加一个
            if not has_return:
                fixed_code += "        return scores\n"
            
            # 添加except子句
            fixed_code += "    except Exception as e:\n"
            fixed_code += "        # Safe fallback that maintains shape\n"
            fixed_code += "        return np.full_like(bins, -1e9, dtype=float) if isinstance(bins, np.ndarray) else np.array([], dtype=float)\n"
        else:
            # 代码已有try-except结构，但可能不完整
            in_try_block = False
            in_except_block = False
            
            for line in code_body.splitlines():
                if line.strip() == "try:":
                    in_try_block = True
                    in_except_block = False
                    fixed_code += "    " + line.strip() + "\n"
                elif line.strip().startswith("except"):
                    in_try_block = False
                    in_except_block = True
                    fixed_code += "    except Exception as e:\n"
                elif in_try_block:
                    fixed_code += "        " + line.strip() + "\n"
                elif in_except_block:
                    fixed_code += "        " + line.strip() + "\n"
                else:
                    fixed_code += "    " + line.strip() + "\n"
            
            # 如果没有except块，添加一个
            if not in_except_block:
                fixed_code += "    except Exception as e:\n"
                fixed_code += "        # Safe fallback that maintains shape\n"
                fixed_code += "        return np.full_like(bins, -1e9, dtype=float) if isinstance(bins, np.ndarray) else np.array([], dtype=float)\n"
        
        # 最终验证
        valid, issues = self._validate_priority_code(fixed_code)
        if not valid:
            logging.warning(f"Advanced LLM: Fix attempt failed, issues remain: {issues}")
            # 如果修复失败，返回安全回退代码
            return self._get_safe_fallback_code()
        
        return fixed_code
    
    def _get_safe_fallback_code(self) -> str:
        """返回一个安全的回退代码，实现简单的Best Fit策略但有完善的错误处理。"""
        fallback_code = """    # Fallback Best-Fit implementation with safe error handling
    try:
        # Convert input if needed
        if not isinstance(bins, np.ndarray):
            try:
                bins = np.array(bins, dtype=float)
            except:
                return np.array([], dtype=float)
        
        # Implement Best-Fit strategy (higher score for smaller remaining space)
        remaining_space = bins - item
        
        # Create scores (negative remaining space = Best Fit heuristic)
        scores = -remaining_space
        
        # Handle any NaN/inf values
        return np.nan_to_num(scores, nan=-1e9, posinf=-1e9, neginf=-1e9)
    except Exception as e:
        # Safe fallback that maintains shape
        return np.full_like(bins, -1e9, dtype=float) if isinstance(bins, np.ndarray) else np.array([], dtype=float)
"""
        return fallback_code
    
    #禁用 BaseLLM 的 fallback 正则提取
    def _fallback_code_extraction(self, text: str) -> str | None:
        logging.debug("Advanced LLM: Fallback regex extraction is disabled.")
        return None