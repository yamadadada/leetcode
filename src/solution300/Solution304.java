package solution300;

public class Solution304 {

    /**
     * 304. 二维区域和检索 - 矩阵不可变
     */
    static class NumMatrix {

        private final int[][] sums;

        public NumMatrix(int[][] matrix) {
            int[][] sums1;
            if (matrix.length == 0) {
                sums1 = new int[1][1];
            } else {
                sums1 = new int[matrix.length + 1][matrix[0].length + 1];
            }
            sums = sums1;
            for (int i = 1; i < matrix.length + 1; i++) {
                for (int j = 1; j < matrix[0].length + 1; j++) {
                    sums[i][j] = sums[i][j - 1] + (sums[i - 1][j] - sums[i - 1][j - 1]) + matrix[i - 1][j - 1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
        }
    }

    public static void main(String[] args) {
        int[][] matrix = new int[5][5];
        matrix[0] = new int[]{3, 0, 1, 4, 2};
        matrix[1] = new int[]{5, 6, 3, 2, 1};
        matrix[2] = new int[]{1, 2, 0, 1, 5};
        matrix[3] = new int[]{4, 1, 0, 1, 7};
        matrix[4] = new int[]{1, 0, 3, 0, 5};
        Solution304 solution = new Solution304();
        Solution304.NumMatrix numMatrix = new Solution304.NumMatrix(matrix);
    }

}
