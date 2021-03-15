package solution0;

import java.util.ArrayList;
import java.util.List;

public class Solution54 {

    /**
     * 54. 螺旋矩阵
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        int top = 0;
        int down = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        List<Integer> ans = new ArrayList<>();
        int i = 0;
        int j = 0;
        ans.add(matrix[i][j]);
        while (top <= down && left <= right) {
            while (top <= down && left <= right && j + 1 <= right) {
                j++;
                ans.add(matrix[i][j]);
            }
            top++;
            while (top <= down && left <= right && i + 1 <= down) {
                i++;
                ans.add(matrix[i][j]);
            }
            right--;
            while (top <= down && left <= right && j - 1 >= left) {
                j--;
                ans.add(matrix[i][j]);
            }
            down--;
            while (top <= down && left <= right && i - 1 >= top) {
                i--;
                ans.add(matrix[i][j]);
            }
            left++;
        }
        return ans;
    }
}
