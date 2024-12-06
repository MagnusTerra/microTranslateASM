from whats_that_code.election import guess_language_all_methods

def detect_language(code_snippet, filename=None):
    try:
        programming_code = guess_language_all_methods(code_snippet)
        
        if programming_code == filename :
            return {
                "status" : True,
                "content" : programming_code
            }
        
        else :
            return {
                "status" : False,
                "content" : f"Mismatch:Detected {programming_code} expected {filename}"
            }
            
    except Exception as e:
        return {f'An unexpecteed error have ocurred: {e}'}
        

# Example Usage
java_code = """
class Test
{
    public static void main(String []args)
    {
        System.out.println("My First Java Program.");
    }
};
"""



result = detect_language(java_code, 'python')

print(result)