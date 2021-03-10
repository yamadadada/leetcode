package solution200;

import java.util.Stack;

public class Solution224 {

    /**
     * 224. 基本计算器
     * @param s
     * @return
     */
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        int a = 1;
        s = s.replace(" ", "");
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                if (i > 0 && s.charAt(i - 1) == '-') {
                    stack.add(-1);
                    a *= -1;
                } else {
                    stack.add(1);
                }
            } else if (s.charAt(i) == ')') {
                int top = stack.pop();
                if (top == -1) {
                    a *= -1;
                }
            } else if (Character.isDigit(s.charAt(i))) {
                int start = i;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    i++;
                }
                int value = Integer.parseInt(s.substring(start, i));
                if (start > 0 && s.charAt(start - 1) == '-') {
                    res += a * -1 * value;
                } else {
                    res += a * value;
                }
                i--;
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Solution224 solution = new Solution224();
        System.out.println(solution.calculate("(7)-(0)+(4)"));
    }
}
