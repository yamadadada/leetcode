package solution800;

public class Solution832 {

    /**
     * 832. 翻转图像
     * @param A
     * @return
     */
    public int[][] flipAndInvertImage(int[][] A) {
        if (A.length == 0) {
            return A;
        }
        int[][] res = new int[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                if (A[i][A[i].length - 1 - j] == 0) {
                    res[i][j] = 1;
                } else {
                    res[i][j] = 0;
                }
            }
        }
        return res;
    }
}
