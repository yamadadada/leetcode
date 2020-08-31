package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class ListUtil {

    /**
     * 通过控制台的输入生成list
     * @return 生成的list
     */
    public static List<List<String>> generate2List() {
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
     * 通过控制台的输入生成list
     * @return 生成的list
     */
    public static List<List<Integer>> generate2IntegerList() {
        List<List<String>> lists = generate2List();
        List<List<Integer>> results = new ArrayList<>();
        for (List<String> list : lists) {
            List<Integer> result = new ArrayList<>();
            for (String s : list) {
                result.add(Integer.parseInt(s));
            }
            results.add(result);
        }
        return results;
    }
}
