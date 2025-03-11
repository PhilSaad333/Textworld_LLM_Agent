from filter_gameplay_data import check_format_command_tags

def test_command_tags():
    """Test various cases of command tag formatting"""
    test_cases = [
        # Valid cases
        ("Simple valid case", "<command>go north</command>", True),
        ("Valid with whitespace", "<command> go north </command>", True),
        ("Valid with newlines", "<command>\ngo north\n</command>", True),
        
        # Invalid cases
        ("No tags", "go north", False),
        ("Only opening tag", "<command>go north", False),
        ("Only closing tag", "go north</command>", False),
        ("Multiple command pairs", "<command>go north</command><command>go south</command>", False),
        ("Nested tags", "<command>go <command>north</command></command>", False),
        ("Wrong order", "</command>go north<command>", False),
        ("Room tags between command tags", "<command>go north<room>Kitchen</room></command>", True),  # This is actually okay
        ("Extra command tags inside", "<command>go north<command>nested</command></command>", False),
    ]
    
    print("Testing command tag validation:")
    print("-" * 50)
    
    for test_name, text, expected in test_cases:
        result = check_format_command_tags(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} {test_name}")
        print(f"   Input: {text}")
        print(f"   Expected: {expected}, Got: {result}")
        print()

if __name__ == "__main__":
    test_command_tags() 