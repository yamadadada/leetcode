package solution400;

public class Solution424 {

    /**
     * 424. 替换后的最长重复字符
     * @param s
     * @param k
     * @return
     */
    public int characterReplacement(String s, int k) {
        int left = 0;
        int right = 0;
        int[] count = new int[26];
        int max = 0;
        while (right < s.length()) {
            count[s.charAt(right) - 'A']++;
            max = Math.max(max, count[s.charAt(right) - 'A']);
            if (right - left + 1 - max > k) {
                count[s.charAt(left) - 'A']--;
                left++;
            }
            right++;
        }
        return right - left;
    }
}
