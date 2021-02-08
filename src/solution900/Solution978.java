package solution900;

public class Solution978 {

    /**
     * 978.最长湍流子数组
     * @param arr
     * @return
     */
    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length;
        int ret = 1;
        int left = 0;
        int right = 0;
        while (right < n - 1) {
            if (left == right) {
                if (arr[left] == arr[left + 1]) {
                    left++;
                }
                right++;
            } else {
                if (arr[right - 1] < arr[right] && arr[right] > arr[right + 1]) {
                    right++;
                } else if (arr[right - 1] > arr[right] && arr[right] < arr[right + 1]) {
                    right++;
                } else {
                    left = right;
                }
            }
            ret = Math.max(ret, right - left + 1);
        }
        return ret;
    }

    public static void main(String[] args) {
        Solution978 solution = new Solution978();
        int[] arr = {9, 4, 2, 10, 7, 8, 8, 1, 9};
        System.out.println(solution.maxTurbulenceSize(arr));
    }
}
