# streamlit_kmeans

KMeans를 이용한 웹대시보드 프로젝트

1. csv 파일 업로드
2. 업로드한 csv 파일을 데이터프레임으로 읽고
3. KMeans 클러스터링을 하기위해, X로 사용할 컬럼을 유저가 설정
4. WCSS 를 확인하기 위한, 그룹의 갯수를 정할 수 있다.
	1개~ 10개
5. 실행버튼을 누르면 엘보우 메소드 차트를 화면에 표시
6. 그룹핑하고 싶은 갯수를 입력
7. 위에서 입력한 그룹의 갯수로 클러스터링하여 결과를 보여준다.
8. 결과를 csv파일로 저장
