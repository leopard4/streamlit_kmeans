import streamlit as st
import os 
import pandas as pd 
from datetime import date, datetime
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# kmeans 를 이용하면 비슷한 그룹으로 묶어서 맞춤 추천을 할수있다.
# 1. csv 파일 업로드 함수정의
# 디렉토리(폴더)명과 파일을 알려주면,
# 해당 디렉토리에 파일을 저장해 주는 함수.
def save_uploaded_file(directory, file) :
    # 1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)

    # 2. 디렉토리가 있으니, 파일을 저장한다.
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

    # 3. 파일 저장이 성공했으니, 화면에 성공했다고 보여주면서 리턴
    return st.success('{} 에 {} 파일이 저장되었습니다.'.format(directory, file.name))
def main() :
    st.title('K-Means 클러스터링')
   
    menu = ['Image', 'CSV', 'About']

    choice = st.sidebar.selectbox('메뉴', menu )

    if choice == 'Image' :
        st.subheader('이미지 파일 업로드')

        file = st.file_uploader('이미지를 업로드 하세요', type=['jpg','jpeg','png'])

        if file is not None :
            # st.text(file.name)
            # st.text(file.size)
            # st.text(file.type)

            # 파일명을 일관성있게, 회사의 파일명 규칙대로 바꾼다.
            # 현재시간을 조합하여 파일명을 만들면, 
            # 유니크하게 파일명을 지을수 있다.

            current_time = datetime.now()
            print(current_time.isoformat().replace(':','_') )
            current_time = current_time.isoformat().replace(':','_')
            print( current_time + '.jpg' )

            file.name = current_time + '.jpg'

            # 바꾼파일명으로, 파일을 서버에 저장한다.
            save_uploaded_file('tmp', file)


            # 파일을 웹 화면에 나오게.
            img = Image.open(file)
            st.image(img)

    elif choice == 'CSV' :
        st.subheader('CSV 파일 업로드')

        file = st.file_uploader('CSV파일 업로드', type=['csv'])
        

        
        # csv 파일은, 판다스로 읽어서 화면에 보여준다.
        df = pd.read_csv(file)
        # 만약 unnamed가 있다면 제거 하라
        if 'Unnamed: 0' in df.columns :
            df.drop(['Unnamed: 0'], axis = 1, inplace = True)

        st.dataframe( df )
        st.info("비어있는 데이터의 갯수를 확인하세요")
        st.dataframe(df.isna().sum())
        st.info("데이터가 비어있다면 제거해야합니다.")
        if st.button("비어있는 데이터를 없애기"):
            df.dropna(inplace= True)
            st.dataframe(df.isna().sum())
        
        
        column_list = df.columns
        selecte_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', column_list)
        # 문자열컬럼을 숫자로 저장
        col_index = []
        for col in selecte_columns :
            col_index.append (df.columns.get_loc(col))
        # 선택한 컬럼으로 액세스
        st.text(selecte_columns)
        X = df[selecte_columns]
        
        st.dataframe(X)
        st.text(col_index)
        st.write(X)
        # 원핫인코딩   
        ct = ColumnTransformer( [ ("encoder", OneHotEncoder(), col_index )], remainder = 'passthrough')
        X = ct.fit_transform( X )
        st.write(X)
        
        

        # # 피처스케일링
        s_scaler_x = StandardScaler()
        s_scaler_x.fit_transform( X )
        m_scaler_x = MinMaxScaler()
        X = m_scaler_x.fit_transform( X )
        st.write(X)
        
        # st.write(selecte_columns)
        # m_scaler_x = MinMaxScaler()
        # selecte_columns = m_scaler_x.fit_transform( selecte_columns )
        # st.write(selecte_columns)
        # 이밑으로 주석은 시행착오를 기록한것이므로 의미없음.
        # 선택한 컬럼중에 객체로된 컬럼이 있는지 확인
        # for col in selecte_columns :
            
            
            # if df[col].dtype == 'O':
            #     st.info(f'글자로된 {col}컬럼을 선택하여 원핫 인코딩을 진행합니다.')



                # #Step1: 모든 문자를 숫자형으로 변환합니다.
                # encoder = LabelEncoder()
                # encoder.fit(df[col])
                # labels = encoder.transform(df[col])
                # #Step2: 2차원 데이터로 변환합니다.
                # labels = labels.reshape(-1, 1)

                # #Step3: One-Hot Encoding 적용합니다.
                # oh_encoder = OneHotEncoder()
                # oh_encoder.fit(labels)
                # oh_labels = oh_encoder.transform(labels)
                # st.write(oh_labels.toarray())
                # st.write(oh_labels.shape)

                # col 컬럼의 숫자로된 인덱스를 가져온다
                # col_index = df.columns.get_loc(col)
                
                # ct = ColumnTransformer( [ ("encoder", OneHotEncoder(), [col_index]) ], remainder = 'passthrough')


                
                # 버튼을 눌렀을때 레이블인코딩 하는코드인데
                # 이상하게 안되서 새로운 변수로 할당해야할듯
                # 버튼 함수의 지역변수 전역변수 문제인듯함.
                # st.info('글자로된 컬럼은 인코딩을 선택해야만 합니다.')
                # if st.button('레이블인코딩') :
                #     encoder = LabelEncoder()
                #     df[col] = encoder.fit_transform(df[col])
                #     st.info('인코딩 결과를 출력합니다.')
                #     st.write(encoder.classes_)
                    
             # 객체로된 컬럼에 카테고리컬 데이터가 2개인지 그이상인지 확인
        

        if st.button('인코딩이 끝났다면 버튼을 눌르세요'):
            st.dataframe(selecte_columns.head(5))
            if len(selecte_columns) != 0 :
                
                # 3. KMeans 클러스터링을 하기위해, X로 사용할 컬럼을 유저가 설정
                X = df[selecte_columns] 
                st.dataframe(X)
                st.subheader('WCSS를 위한 클러스터링 갯수를 선택')

                #4. WCSS 를 확인하기 위한, 그룹의 갯수를 정할 수 있다.
                #    1개~ 10개
                # 5. 실행버튼을 누르면 엘보우 메소드 차트를 화면에 표시
                max_number = st.slider('최대 그룹 선택', 2, 10, value=10)

                wcss = []
                for k in np.arange(1, max_number+1) :
                    kmeans = KMeans(n_clusters = k , random_state=5)
                    kmeans.fit(X)
                    wcss.append( kmeans.inertia_)
            
                fig1 = plt.figure()
                x = np.arange(1, max_number+1)
                plt.plot(x, wcss)
                plt.title("The Elbow Method")
                plt.xlabel('Number of Clusters')
                plt.ylabel("WCSS")
                st.pyplot(fig1)

                # 실제로 그룹핑할 갯수 선택!
                # k = st.slider("그룹 갯수를 결정하세요",1 , max_number,  value=10 )
                k = st.number_input('그룹 갯수 결정', 1 , max_number)

                kmeans = KMeans(n_clusters = k , random_state =5 )

                # 결정한 갯수를 학습
                y_pred = kmeans.fit_predict(X)
                
                df["Group"] = y_pred 
                
                # 그룹으로 정렬해서 보여줘라
                st.dataframe(df.sort_values('Group'))

                df.to_csv('result.csv')
                
                
                # 화면에 카피가 가능한 코드블럭을 생성해준다.
                # st.write(wcss)
            
                
            

    elif choice == 'About' :
        st.subheader('파일 업로드 프로젝트 입니다.')

    




if __name__ == '__main__' :
    main()