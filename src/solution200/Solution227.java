package solution200;

import java.util.Stack;

public class Solution227 {

    /**
     * 227. 基本计算器 II
     * @param s
     * @return
     */
    public int calculate(String s) {
        s = s.replace(" ", "");
        Stack<Integer> stack = new Stack<>();
        int i = 0;
        while (i < s.length()) {
            if (Character.isDigit(s.charAt(i))) {
                int start = i;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    ++i;
                }
                int value = Integer.parseInt(s.substring(start, i));
                if (start == 0 || s.charAt(start - 1) == '+') {
                    stack.push(value);
                } else if (s.charAt(start - 1) == '-') {
                    stack.push(-value);
                } else if (s.charAt(start - 1) == '*') {
                    int topValue = stack.pop();
                    stack.push(topValue * value);
                } else if (s.charAt(start - 1) == '/') {
                    int topValue = stack.pop();
                    stack.push(topValue / value);
                }
            } else {
                ++i;
            }
        }
        int res = 0;
        for (Integer a : stack) {
            res += a;
        }
        return res;
    }
}
