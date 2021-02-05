package solution1200;

public class Solution1208 {

    /**
     * 1208. 尽可能使字符串相等
     * @param s
     * @param t
     * @param maxCost
     * @return
     */
    public int equalSubstring(String s, String t, int maxCost) {
        int[] nums = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            nums[i] = Math.abs(s.charAt(i) - t.charAt(i));
        }
        int left = 0;
        int right = 0;
        int cost = 0;
        int maxSize = 0;
        while (right < s.length()) {
            cost += nums[right];
            while (cost > maxCost) {
                cost -= nums[left];
                left++;
            }
            maxSize = Math.max(maxSize, right - left + 1);
            right++;
        }
        return maxSize;
    }
}
