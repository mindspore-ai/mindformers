# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Test deprecation logs of deprecated API.
How to run this:
pytest tests/st/test_logs/test_deprecation_logs.py
"""
import inspect
import warnings
from functools import wraps
from unittest.mock import patch

import pytest

from mindformers.utils import deprecated


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestDeprecationLogs:
    """A test class for testing deprecation logs."""
    def test_basic_deprecation_without_args(self):
        """
        Test basic deprecation functionality without any parameters
        Expected: Display default deprecation warning
        """
        @deprecated()
        def old_function():
            return "function called"

        with pytest.warns(DeprecationWarning) as record:
            result = old_function()

        assert result == "function called"
        assert len(record) == 1
        assert "deprecated" in str(record[0].message).lower()

    def test_deprecation_with_reason(self):
        """
        Test decorator with deprecation reason
        Expected: Warning message contains the specified reason
        """
        reason = "Use new_function() instead"

        @deprecated(reason=reason)
        def old_function():
            return "function called"

        with pytest.warns(DeprecationWarning) as record:
            result = old_function()

        assert result == "function called"
        assert reason in str(record[0].message)

    def test_deprecation_with_version(self):
        """
        Test decorator with version information
        Expected: Warning message contains version information
        """
        version = "2.0.0"

        @deprecated(version=version)
        def old_function():
            return "function called"

        with pytest.warns(DeprecationWarning) as record:
            result = old_function()

        assert result == "function called"
        assert version in str(record[0].message)

    def test_deprecation_with_all_arguments(self):
        """
        Test decorator with all parameters (reason and version)
        Expected: Warning message contains both reason and version information
        """
        reason = "Use new API"
        version = "3.0.0"

        @deprecated(reason=reason, version=version)
        def old_function():
            return "function called"

        with pytest.warns(DeprecationWarning) as record:
            result = old_function()

        assert result == "function called"
        message = str(record[0].message)
        assert reason in message
        assert version in message

    def test_preserve_function_signature(self):
        """
        Test whether the decorator preserves the original function signature
        Expected: Decorated function signature is identical to the original function
        """
        @deprecated(reason="Test signature preservation")
        def function_with_args(a, b, c=3, *args, **kwargs):
            """Test function docstring"""
            return a + b + c

        # Check if signature remains unchanged
        original_sig = inspect.signature(function_with_args)
        assert str(original_sig) == "(a, b, c=3, *args, **kwargs)"

        # Check if function name remains unchanged
        assert function_with_args.__name__ == "function_with_args"

        # Check if docstring remains unchanged
        assert function_with_args.__doc__ == "Test function docstring"

    def test_preserve_function_attributes(self):
        """
        Test whether the decorator preserves other function attributes
        Expected: Decorated function retains the original function's attributes
        """
        @deprecated()
        def test_func():
            pass

        # Add custom attribute
        test_func.custom_attr = "test_value"

        # Verify attribute is preserved
        assert hasattr(test_func, "custom_attr")
        assert test_func.custom_attr == "test_value"

    def test_multiple_calls_generate_warning_each_time(self):
        """
        Test whether multiple function calls generate warnings each time
        Expected: Each call generates a deprecation warning
        """
        @deprecated(reason="Test multiple warnings")
        def test_func():
            return "called"

        # First call
        with pytest.warns(DeprecationWarning) as record1:
            test_func()

        # Second call
        with pytest.warns(DeprecationWarning) as record2:
            test_func()

        assert len(record1) == 1
        assert len(record2) == 1

    def test_filter_deprecation_warnings(self):
        """
        Test whether deprecation warnings can be filtered
        Expected: When warnings are filtered, no warning is triggered
        """
        @deprecated(reason="Test warning filtering")
        def test_func():
            return "called"

        # Ignore deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = test_func()

        assert result == "called"

    def test_complex_functionality(self):
        """
        Test complex functions
        Expected: Complex function executes normally and generates warnings
        """
        @deprecated(reason="Test complex function", version="1.5.0")
        def complex_function(data, multiplier=2, **kwargs):
            """Process complex data"""
            result = sum(data) * multiplier
            if kwargs.get("verbose"):
                print(f"Processing {len(data)} items")
            return result

        with pytest.warns(DeprecationWarning) as record:
            result = complex_function([1, 2, 3], multiplier=3, verbose=False)

        assert result == 18  # (1+2+3) * 3
        assert "Test complex function" in str(record[0].message)
        assert "1.5.0" in str(record[0].message)

    def test_method_deprecation(self):
        """
        Test deprecation of class methods
        Expected: Class methods are correctly deprecated and generate warnings
        """
        class TestClass:
            @deprecated(reason="Use new_method instead", version="2.0.0")
            def old_method(self, x):
                return x * 2

            @classmethod
            @deprecated(reason="Class method deprecated")
            def old_classmethod(cls, x):
                return x * 3

            @staticmethod
            @deprecated(reason="Static method deprecated", version="1.0.0")
            def old_staticmethod(x):
                return x * 4

        obj = TestClass()

        # Test instance method
        with pytest.warns(DeprecationWarning) as record:
            assert obj.old_method(5) == 10
        assert "Use new_method instead" in str(record[0].message)
        assert "2.0.0" in str(record[0].message)

        # Test class method
        with pytest.warns(DeprecationWarning) as record:
            assert TestClass.old_classmethod(5) == 15
        assert "Class method deprecated" in str(record[0].message)

        # Test static method
        with pytest.warns(DeprecationWarning) as record:
            assert TestClass.old_staticmethod(5) == 20
        assert "Static method deprecated" in str(record[0].message)
        assert "1.0.0" in str(record[0].message)

    def test_nested_decorators(self):
        """
        Test nesting with other decorators
        Expected: Deprecation decorator works correctly with other decorators
        """
        def mock_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) + "_decorated"
            return wrapper

        @mock_decorator
        @deprecated(reason="Test nested decorators")
        def nested_function():
            return "base"

        with pytest.warns(DeprecationWarning):
            result = nested_function()

        assert result == "base_decorated"

    def test_argument_passing(self):
        """
        Test correct argument passing
        Expected: All arguments are correctly passed to the original function
        """
        @deprecated(reason="Test argument passing")
        def function_with_args(*args, **kwargs):
            return {
                'args': args,
                'kwargs': kwargs
            }

        with pytest.warns(DeprecationWarning):
            result = function_with_args(1, 2, 3, a=4, b=5)

        assert result['args'] == (1, 2, 3)
        assert result['kwargs'] == {'a': 4, 'b': 5}

    def test_return_value_preservation(self):
        """
        Test correct return value passing
        Expected: Original function's return value is correctly returned
        """
        expected_return = [1, 2, 3, 4, 5]

        @deprecated(reason="Test return value")
        def return_list():
            return expected_return

        with pytest.warns(DeprecationWarning):
            result = return_list()

        assert result == expected_return
        assert result is expected_return  # Verify it's the same object

    def test_exception_propagation(self):
        """
        Test correct exception propagation
        Expected: Exceptions raised by the original function are correctly propagated
        """
        @deprecated(reason="Test exception")
        def function_that_raises():
            raise ValueError("Test exception")

        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="Test exception"):
                function_that_raises()

    @patch('warnings.warn')
    def test_warning_format(self, mock_warn):
        """
        Test warning message format
        Expected: Warning message contains the correct format
        """
        reason = "Custom reason"
        version = "1.0.0"

        @deprecated(reason=reason, version=version)
        def test_func():
            pass

        test_func()

        # Verify warnings.warn is called correctly
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args[0][0]  # Get warning message
        assert reason in str(call_args)
        assert version in str(call_args)
        assert "deprecated" in str(call_args).lower()
