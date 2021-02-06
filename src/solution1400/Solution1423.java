package solution1400;

public class Solution1423 {

    /**
     * 1423. 可获得的最大点数
     * @param cardPoints
     * @param k
     * @return
     */
    public int maxScore(int[] cardPoints, int k) {
        // 反向思维，长度为n-k的滑动窗口
        int length = cardPoints.length - k;
        int sum = 0;
        for (int i = 0; i < length; i++) {
            sum += cardPoints[i];
        }
        int all = sum;
        int min = all;
        int right = length;
        while (right < cardPoints.length) {
            all += cardPoints[right];
            sum += cardPoints[right];
            sum -= cardPoints[right - length];
            min = Math.min(min, sum);
            right++;
        }
        return all - min;
    }
}
