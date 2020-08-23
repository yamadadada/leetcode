import utils.ArrayUtil;

public class Test {

    public static void main(String[] args) {
        Solution solution = new Solution();
        char[][] c = ArrayUtil.generate2CharArray();
        char[][] board = solution.updateBoard(c, new int[]{1, 2});
        ArrayUtil.print2CharArray(board);
    }
}
