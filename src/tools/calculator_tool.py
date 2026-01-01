from typing import Union
from langchain.tools import BaseTool
import numpy as np

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform mathematical calculations"

    def __init__(self):
        super().__init__()
       
    def _run(self, expression: str) -> Union[float, str]:
        """Evaluate a mathematical expression"""
        try:
            # Create a safe dictionary of allowed mathematical functions
            safe_dict = {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'pow': pow,
                'sqrt': np.sqrt,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'pi': np.pi,
                'e': np.e
            }
            
            # Replace common mathematical notations
            expression = expression.replace('^', '**')
            
            # Evaluate the expression in a safe environment
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
            
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    async def _arun(self, expression: str) -> Union[float, str]:
        """Async implementation of run"""
        return self._run(expression)
