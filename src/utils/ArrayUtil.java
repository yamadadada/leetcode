package utils;

import java.util.*;

public class ArrayUtil {

    /**
     * 通过控制台的输入生成list
     * @return 生成的list
     */
    public static List<List<String>> generateList() {
        Scanner sc = new Scanner(System.in);
        List<List<String>> result = new ArrayList<>();
        System.out.println("请输入数组元素，以空格分割，输入exit退出：");
        while (true) {
            String line = sc.nextLine();
            if ("exit".equals(line)) {
                break;
            }
            String[] arr = line.split("\\s+");
            result.add(new ArrayList<>(Arrays.asList(arr)));
        }
        return result;
    }

    /**
     * 通过控制台的输入返回一维String数组
     * @return 生成的一维String数组
     */
    public static String[] generateStringArray() {
        List<List<String>> lists = generateList();
        return (String[]) lists.get(0).toArray();
    }

    /**
     * 通过控制台的输入返回二维String数组
     * @return 生成的二维String数组
     */
    public static String[][] generate2StringArray() {
        List<List<String>> lists = generateList();
        String[][] array = new String[lists.size()][];
        for (int i = 0; i < lists.size(); ++i) {
            array[i] = lists.get(i).toArray(new String[0]);
        }
        return array;
    }

    /**
     * 通过控制台的输入返回一维char数组
     * @return 生成的一维char数组
     */
    public static char[] generateCharArray() {
        String[] strings = generateStringArray();
        char[] chars = new char[strings.length];
        for (int i = 0; i < strings.length; ++i) {
            chars[i] = strings[i].charAt(0);
        }
        return chars;
    }

    /**
     * 通过控制台的输入返回二维char数组
     * @return 生成的二维char数组
     */
    public static char[][] generate2CharArray() {
        String[][] strings = generate2StringArray();
        char[][] chars = new char[strings.length][];
        for (int i = 0; i < strings.length; ++i) {
            chars[i] = new char[strings[i].length];
            for (int j = 0; j < strings[i].length; ++j) {
                chars[i][j] = strings[i][j].charAt(0);
            }
        }
        return chars;
    }

    /**
     * 控制台输出一维数组
     * @param array 一维数组
     */
    public static void printArray(Object[] array) {
        for (Object a : array) {
            System.out.print(a + " ");
        }
        System.out.println();
    }

    /**
     * 控制台输出二维数组
     * @param array 二维数组
     */
    public static void print2Array(Object[][] array) {
        for (Object[] arr : array) {
            printArray(arr);
        }
    }

    /**
     * 控制台输出一维char数组
     * @param array 一维char数组
     */
    public static void printCharArray(char[] array) {
        for (char a : array) {
            System.out.print(a + " ");
        }
        System.out.println();
    }

    /**
     * 控制台输出二维数组
     * @param array 二维数组
     */
    public static void print2CharArray(char[][] array) {
        for (char[] arr : array) {
            printCharArray(arr);
        }
    }
}
