package solution500;

public class Solution567 {

    /**
     * 567. 字符串的排列
     * @param s1
     * @param s2
     * @return
     */
    public boolean checkInclusion(String s1, String s2) {
        int[] nums1 = new int[26];
        int[] nums2 = new int[26];
        for (char c : s1.toCharArray()) {
            nums1[c - 'a']++;
        }
        int count1 = 0;
        int count2 = 0;
        for (int num : nums1) {
            if (num > 0) {
                count1++;
            }
        }
        int left = 0;
        int right = 0;
        while (right < s2.length()) {
            if (nums1[s2.charAt(right) - 'a'] > 0) {
                nums2[s2.charAt(right) - 'a']++;
                if (nums2[s2.charAt(right) - 'a'] == nums1[s2.charAt(right) - 'a']) {
                    count2++;
                }
            }

            while (count1 == count2) {
                if (right - left + 1 == s1.length()) {
                    return true;
                }
                if (nums1[s2.charAt(left) - 'a'] > 0) {
                    nums2[s2.charAt(left) - 'a']--;
                    if (nums2[s2.charAt(left) - 'a'] < nums1[s2.charAt(left) - 'a']) {
                        count2--;
                    }
                }
                left++;
            }
            right++;
        }
        return false;
    }
}
