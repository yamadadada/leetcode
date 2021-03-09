package solution1000;

import java.util.Stack;

public class Solution1047 {

    /**
     * 1047. 删除字符串中的所有相邻重复项
     * @param S
     * @return
     */
    public String removeDuplicates(String S) {
        Stack<Character> stack = new Stack<>();
        for (char c : S.toCharArray()) {
            if (!stack.isEmpty() && stack.peek() == c) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Character c :stack) {
            sb.append(c);
        }
        return sb.toString();
    }
}
