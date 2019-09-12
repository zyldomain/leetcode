import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		int m ,n;
		Scanner scanner = new Scanner(System.in);
		m = scanner.nextInt();
		n = scanner.nextInt();
		int[] array1 = new int[n];
		int[] array2 = new int[n];
		int[] array3 = new int[n];
		for(int i = 0 ; i < n ; i++) {
			array1[i] = scanner.nextInt();
		}
		for(int i = 0 ; i < n ; i++) {
			array2[i] = scanner.nextInt();
		}
		boolean[] used1 = new boolean[n];
		boolean[] used2 = new boolean[n];
		int[][] dp = new int[n][n];
		for(int i = 0 ; i< n ; i++) {
			for(int j = 0; j < n ; j++) {
				dp[i][j] = (array1[i] + array2[j]) % m;
			}
		}
		int count = 0;
		int k = 0;
		while(count < n) {
			int p = 0,q = 0,max = -1;
			for(int i = 0 ; i < n; i++) {
				if(used1[i]) {
					continue;
				}
				for(int j = 0; j < n; j++) {
					if(used2[j]) {
						continue;
					}
					if(max < dp[i][j]) {
						max = dp[i][j];
						p = i;
						q = j;
					}
				}
			}
			used1[p] = true;
			used2[q] = true;
			count++;
			array3[k++] = max;
		}
		for(int p = 0 ; p < n ; p++) {
			System.out.print(array3[p] + " ");
		}
	}
}


