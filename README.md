# MovieGenreClassification-INT3405-project-
Movie Genre Classification Báo Cáo Cuối Kỳ Môn Học Máy Nhóm 13 (INT3405-project) 
Movie Genre Classification
Báo Cáo Cuối Kỳ Môn Học Máy Nhóm 13 (INT3405-project) 
 
I.	INTRODUCTION (HEADING 1)
Trong dự án này, chúng ta đặt ra nhiệm vụ xây dựng một mô hình học máy nhằm phân loại thể loại của các bộ phim dựa trên các thông tin đa dạng bao gồm tiêu đề phim, ảnh bìa, và xếp hạng từ người dùng. Nhiệm vụ này có ý nghĩa quan trọng trong lĩnh vực đề xuất nội dung và tìm kiếm phim, giúp cải thiện trải nghiệm người dùng. 
Dữ liệu được sử dụng trong dự án được trích xuất từ tập dữ liệu Movelens 1M, một nguồn tài nguyên quan trọng trong lĩnh vực đánh giá phim và lọc cộng tác. Dữ liệu này đã được chia thành tập huấn luyện và tập kiểm tra, giúp đảm bảo mô hình có khả năng tổng quát hóa.
Nhiệm vụ phân loại thể loại phim đặt ra những thách thức động với sự đa dạng của dữ liệu, sự không đồng nhất của tiêu đề phim và sự đa dạng của các thể loại. Đồng thời, việc tích hợp thông tin từ các nguồn khác nhau như ảnh và xếp hạng người dùng làm tăng tính phức tạp của mô hình.
Mục tiêu chính của dự án là xây dựng một mô hình có khả năng phân loại thể loại phim với hiệu suất cao trên tập kiểm tra, đồng thời đảm bảo tính tổng quát hóa và khả năng ứng dụng thực tế.
Báo cáo này sẽ được chia thành các phần chính như sau:
•	Phần I: Giới Thiệu (Hiện tại bạn đang đọc)
•	Phần II: Xử Lý Dữ Liệu và Tiền Xử Lý
•	Phần III: Xây Dựng Mô Hình, Đào Tạo
 	và Đánh Giá Mô Hình
•	Phần IV: Kết Luận và Tương Lai 
/ Nhận Xét và Cải Thiện:
Chúng ta sẽ bắt đầu với Phần II, nơi chúng ta sẽ khám phá quá trình xử lý dữ liệu và tiền xử lý trước khi xây dựng mô hình.
II.	XỬ LÝ DỮ LIỆU VÀ TIỀN XỬ LÝ
A.	Nguồn dữ liệu (Heading 2)

Nhóm sử dụng dữ liệu gốc được cung cấp từ : 
https://drive.google.com/uc?id=1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD
tập dữ liệu Movelens 1M này bao gồm các thông tin như tiêu đề phim, ảnh bìa, xếp hạng người dung, thông tin người dung đã đánh giá ( giới tính, tuổi, nghề nghiệp ) và thể loại. Dữ liệu này sẽ được chia thành tập huấn luyện và tập kiểm tra để huấn luyện và đánh giá mô hình. : 
ml1m-image: thư mục chứa 3952 ảnh bìa/poster của các bộ phim, tên ảnh là id của các bộ phim mà nó liên kết, genres.txt: một tệp văn bản chứa tất cả các thể loại trong cả dữ liệu huấn luyện và kiểm tra, movies_{train|test}.dat: tệp đào tạo và thử nghiệm, chứa thông tin về phim, ví dụ: id, tên và thể loại của phim, rating.dat: file dữ liệu chứa xếp hạng của các cặp phim người dùng., user.dat: file dữ liệu chứa thông tin của người dùng.
B.	mã hóa thể loại

Không thể sử dụng thuật toán phân loại được giám sát trực tiếp trên tập dữ liệu nhiều nhãn Do đó, trước tiên chúng ta sẽ phải chuyển đổi biến mục tiêu của mình sử dụng vector nhị phân mã hóa nhiều thể loại bằng cách sử dụng tập dữ liệu giả Ví dụ :.
 
x	y
x1	[t1,t5]
x2	[t1,t2,t3]
x3	[t4]

x	t1	t2	t3	t4	t5
x1	1	0	0	0	1
x2	1	1	1	0	0
x3	0	0	0	1	0
		
Ở đây, x và y lần lượt là đặc trưng và giá trị đích (nhãn) - đó là tập dữ liệu nhiều nhãn. Chúng ta sẽ sử dụng Binary Relevance để chuyển đổi 
Ví dụ, trong trường hợp mã hóa thể loại : giả sử phim có ba thể loại: Hành động, Kịch tính, và Hài phim thuộc cả hai thể loại Hành động và Hài, vector được mã hóa có thể trông như sau: v( 1; 0; 1) Hành động: 1 Kịch tính: 0 Hài: 1

C.	Tải dữ liệu, thêm thư viện, tạo datafame

Phần này chúng ta phải import các thư viện cần thiết , tải dữ liệu ghép dữ liệu theo id tạo datafame, giữ các đặc trưng cần sử dụng chuẩn bị để xử lý
movieid 	title	genre	id	img_path
1	185		Net	[Sci-Fi]	185	./1650.jpg
D.	Tiền xử lý dữ liệu hình ảnh

Trong phần này, chúng ta sẽ tập trung vào xử lý dữ liệu hình ảnh của bộ dữ liệu phim, đảm bảo rằng chúng ta có thể tích hợp nó vào mô hình của mình một cách hiệu quả.
Chúng ta sử dụng thư viện OpenCV để đọc và xử lý ảnh. Hàm preprocess_image thực hiện các bước như resize ảnh và chuẩn hóa giá trị pixel.
Đọc dữ liệu phim từ tập tin và tạo đường dẫn đầy đủ cho ảnh từ tên tệp.
Áp dụng hàm preprocess_image cho toàn bộ tập dữ liệu hình ảnh của cả tập huấn luyện và tập kiểm tra.
Bây giờ, dữ liệu hình ảnh đã được chuẩn bị và có thể được tích hợp vào mô hình. Phần tiếp theo của báo cáo có thể tập trung vào xử lý tiêu đề phim và thể loại.
E.	Tiền xử lý tiêu đề phim 

Trong phần này, chúng ta sẽ thảo luận về cách xử lý dữ liệu văn bản từ tiêu đề phim để có thể tích hợp nó vào mô hình học máy, các bước :
Tiền xử lý tiêu đề phim : Chúng ta sử dụng một số hàm tiền xử lý cơ bản để chuyển đổi các tiêu đề phim thành biểu diễn số có thể sử dụng trong mô hình : tách các từ trong tiêu đề, loại bỏ các ký tự không mong muốn, chuyển đổi về chữ thường.
Tạo Từ Vựng: tạo một từ vựng từ tất cả các từ xuất hiện trong tiêu đề phim bằng hàm . create_vocab()
Tạo Biểu Diễn Số cho Mỗi Tiêu Đề Phim: tạo một biểu diễn số cho mỗi tiêu đề phim bằng cách sử dụng một vectơ nhị phân để biểu diễn việc xuất hiện của từng từ trong từ vựng.
F.	Tiền xử lý điểm đánh giá và thông tin người xem 

Đầu tiên, chọn các đặc trưng cần sử dụng kiểm tra : 'gender', 'rating', 'age', 'occupation'
Tiếp theo, xử lý giá trị NaN, và xóa các dòng đó
Cuối cùng là chuẩn bị dữ liệu với PyTorch bằng cách chuyển đổi dữ liệu train test thành tensor PyTorch
III.	XÂY DỰNG MÔ HÌNH
Before you begin to format your paper, first write and save the content as a separate text file. Complete all content and organizational editing before formatting. Please note sections A-D below for more information on proofreading, spelling and grammar.
Keep your text and graphic files separate until after the text has been formatted and styled. Do not use hard tabs, and limit use of hard returns to only one return at the end of a paragraph. Do not add any kind of pagination anywhere in the paper. Do not number text heads-the template will do that for you.
A.	 DenseNet 

Define abbreviations and acronyms the first time they are used in the text, even after they have been defined in the abstract. Abbreviations such as IEEE, SI, MKS, CGS, sc, dc, and rms do not have to be defined. Do not use abbreviations in the title or heads unless they are unavoidable.
1)	Giới thiệu

DenseNet (Densely Connected Convolutional Networks) là một kiến trúc mạng nơ-ron sử dụng trong lĩnh vực thị giác máy tính, đặc biệt là trong bài toán phân loại hình ảnh.. DenseNet sử dụng kiến trúc kết nối mật độ cao, trong đó mỗi lớp đầu ra được kết nối với tất cả các lớp đầu vào trước đó. Điều này giúp kiến trúc trở nên rất "dày đặc" với thông tin và giúp giảm độ mất mát thông tin trong quá trình lan truyền ngược. Do tính chất kết nối mật độ cao, DenseNet có xu hướng sử dụng số lượng tham số thấp hơn so với các mô hình khác như VGG hay ResNet, trong khi vẫn duy trì hiệu suất cao.
DenseNet đã được huấn luyện trên tập dữ liệu ImageNet lớn đã học được các đặc trưng cấp cao và có khả năng tổng quát hóa tốt. Đồng thời nó còn có khả năng transfer learning tốt tiết kiệm thời gian và nguồn lực đào tạo trên tập dữ liệu của mình. Hơn hết Kết Nối Mật Độ Cao và Đặc Trưng Tái Sử Dụng: DenseNet sử dụng kết nối mật độ cao, giúp mô hình học được các biểu diễn đặc trưng chung từ poster phim. Điều này đặc biệt hữu ích khi poster thường chứa nhiều thông tin đa dạng về nội dung và phong cách.Với những đặc điểm trên, DenseNet121 là một lựa chọn mạnh mẽ cho bài toán phân loại thể loại phim dựa trên poster
2)	 Hoat động 

 Hê thống cho các hình ảnh RGB ba chiều ban đầu được truyền qua một chuỗi gồm lớp chập, lớp chuẩn hóa hàng loạt, ReLU và lớp gộp. Đầu ra của chúng được truyền qua mạng gồm bốn khối dày đặc được kết nối thông qua các lớp chuyển tiếp. Một khối dày đặc chứa một chuỗi các lớp dày đặc bao gồm lớp chuẩn hóa lô, ReLU và lớp chập. Lớp chuyển tiếp bao gồm lớp chuẩn hóa hàng loạt và ReLU, theo sau là lớp tích chập và lớp gộp. Cuối cùng, dữ liệu đầu ra của vectơ  được truyền qua lớp kết nối đầy đủ .Thêm nữa, Perceptron nhiều lớp ở cuối lớp chập hiện bao gồm 1024 đặc trưng đầu vào, với hàm sigmoid giúp mô hình dự đoán tốt các giá trị đích( thể loại) nhiều nhãn(18 thể loại).
 
Fig. 1.	Sơ đồ DenseNet121 Model
	DenseNet121 là một mạng nơ-ron sâu bao gồm các lớp chập (convolutional layers), lớp kích hoạt (activation layers), và các lớp tối ưu hóa. Mô hình được chia thành các khối (blocks) gọi là "dense blocks," và giữa các dense block là các lớp "transition" để giảm kích thước của đặc trưng. Mỗi dense block bao gồm một chuỗi các lớp chập (convolutional layer) và lớp kích hoạt (activation layer). Các lớp này thường được xếp chồng lên nhau. Kết nối mật độ cao trong dense block giúp giảm độ mất mát thông tin và tăng cường khả năng học các biểu diễn phức tạp.
	Sau mỗi dense block, một lớp "transition" được thêm vào để giảm kích thước của đặc trưng và làm giảm độ sâu của mạng. Transition layer thường bao gồm một lớp chập 1x1 để giảm số lượng kênh đặc trưng và một lớp pooling để giảm kích thước không gian. 
	Khối linear thực hiện phép biến đổi tuyến tính trên đầu vào bằng cách sử dụng ma trận trọng số và thêm vector độ chệch (bias). Sử dụng để học các mối quan hệ tuyến tính giữa đầu vào và đầu ra.
	Lớp BatchNorm là chuẩn hóa đầu vào của, đảm bảo rằng giá trị đầu vào có phân phối trung bình gần 0 và độ lệch chuẩn gần 1. Điều này giúp mô hình học nhanh hơn và ổn định hơn.
	Khối sigmoid được sử dụng ở lớp đầu ra của mô hình phân loại nhị phân để ánh xạ đầu ra thành xác suất thuộc một lớp.
3)	Loss Fuction và tối ưu hóa

	Sử dụng hàm mất mát BCEWithLogitsLoss, được thiết kế đặc biệt cho bài toán phân loại nhị phân với đầu ra không giới hạn và sử dụng trình tối ưu hóa Adam để thay đổi tốc độ học trong thực hiện gradient descent
4)	Quy Trình Huấn Luyện

Có thể huấn luyện mô hình trên số lượng epochs tùy chỉnh đã cho. Dữ liệu huấn luyện được lấy từ train_loader. Trong quá trình huấn luyện, mỗi batch được đưa vào mô hình, gradient được cập nhật và hiển thị thông tin đào tạo như mất mát.
testAccuracy(): Kiểm thử mô hình trên tập kiểm tra và tính toán độ chính xác. Kết quả được in ra cho từng thể loại và tổng thể.

B.	Mô hình neural network phân loại dựa vào đánh giá và thông tin người xem

Trong bài một mô hình phân loại đa nhãn sử dụng mạng neural network đa lớp cũng được đào tạo trên tập dữ liệu huấn luyện. Dưới đây là phân tích về kiến trúc, hoạt động của mô hình và cách nó được đào tạo:

1)	Kiến Trúc Mô Hình:

Mô hình của bạn là một mạng neural network đa lớp với các tầng tuyến tính (fully connected layers) xen kẽ với các hàm kích hoạt ReLU.
Số lượng đầu vào là xác định bởi số lượng đặc trưng bạn đã chọn (gender, rating, age, occupation), và số lượng đầu ra là số lượng thể loại phim.

2)	Hàm Kích Hoạt:

Trong tầng đầu ra, bạn đã sử dụng hàm kích hoạt Sigmoid. Hàm này được sử dụng để chuyển đổi giá trị đầu ra thành xác suất cho mỗi thể loại. Mỗi giá trị đầu ra tương ứng với xác suất thuộc về mỗi thể loại.

3)	Loss Function:

Sử dụng hàm mất mát BCELoss (Binary Cross Entropy Loss) dành cho bài toán phân loại nhị phân nhiều nhãn (multi-label classification). Điều này là lựa chọn phù hợp cho bài toán phân loại thể loại phim.

4)	Quy Trình Huấn Luyện:

Bạn đã sử dụng tối ưu hóa Adam để điều chỉnh trọng số mô hình.
Mô hình được huấn luyện qua nhiều epochs, và mỗi epoch gồm các bước: forward pass, tính toán loss, backward pass và cập nhật trọng số.
Đối với dự đoán trên tập kiểm tra, bạn đã sử dụng ngưỡng 0.3 để chuyển đổi giá trị xác suất thành dự đoán nhãn.

5)	Đánh Giá Mô Hình:

Mô hình có đạt được độ chính xác (accuracy) trên tập kiểm tra là khoảng 22.59% nhưng còn thiếu ổn định
Đối với từng nhãn, kết quả được mô tả trong bảng phân loại (classification report). Bảng này cung cấp thông tin về precision, recall và f1-score cho từng nhãn.

6)	Nhận Xét và Cải Thiện:

Mô hình có thể đối mặt với khó khăn trong việc học các mối quan hệ phức tạp giữa đặc trưng và thể loại phim.
Có thể cần xem xét lại kiến trúc mô hình, thử nghiệm với các hàm kích hoạt và thay đổi tham số đào tạo để cải thiện hiệu suất.
IV.	ĐÁNH GIÁ MÔ HÌNH.
A.	DenseNet 

Mô hình được xây dựng dựa trên kiến trúc DenseNet-121, đã được huấn luyện trước độ sâu (pre-trained) trên tập dữ liệu lớn. Mục tiêu của mô hình là dự đoán nhiều nhãn thể loại phim cho mỗi hình ảnh.
Mô hình đạt được độ chính xác trung bình khoảng 37.8% trên tập kiểm tra sau 3 epochs. Mặc dù kết quả này có vẻ thấp, có thể có nhiều nguyên nhân như:
•	Kích thước dữ liệu nhỏ.
•	Phân phối không cân bằng giữa các nhãn thể loại.
•	Siêu tham số chưa được điều chỉnh tối ưu.
Mô hình có độ chính xác tốt đối với một số thể loại như "Sci-Fi", "Adventure"  Một số thể loại như "Documentary", "Film-Noir", "Crime", "Mystery" có độ chính xác rất thấp gần như  0%.
Mô hình có khả năng đối với một số hình ảnh thử nghiệm, nhưng còn có những sai sót đáng kể.
 
Fig. 2.	Độ chính xác DenseNet. 

 
Fig. 3.	Độ mất mát loss DenseNet. 

B.	Mô hình neural network phân loại dựa vào đánh giá và thông tin người xem

Mô hình có đạt được độ chính xác (accuracy) trên tập kiểm tra là khoảng 22.59% 

Đối với từng nhãn, kết quả được mô tả trong bảng phân loại (classification report). Bảng này cung cấp thông tin về precision, recall và f1-score cho từng nhãn.
 
Fig. 4.	Kết quả khảo sát mô hình neural network
Link file code báo cáo được nhóm lưu trữ trên  :
•	DenseNet: : https://github.com/dh266/Beach_House/blob/main/genre_classification_image1.ipynb
Hoặc: https://colab.research.google.com/drive/1XMlCe7Gjg-cU210T6bgzFKzrlBsP2H4_?usp=sharing

•	Neural network : https://github.com/dh266/Beach_House/blob/main/INT3405E_Group_Project_Modelratinguser.ipynb
Hoặc: https://colab.research.google.com/drive/1ARxB9_Wnh6MSxLBX1pRXNAveGaupnVpy?usp=sharing


TÀI LIỆU THAM KHẢO

[1]	Samuel Sung, Rahul Chokshi “Classification of Movie Posters to Movie Genres,” CS230: Deep Learning, Winter 2018, Stanford University, CA. (LateX template borrowed from NIPS 2017.
[2]	Isaac Rodríguez-Bribiesca and A. Pastor López-Monroy, “Multimodal Weighted Fusion of Transformers for Movie Genre Classification”, Rodríguez Bribiesca et al., maiworkshop 2021)
[3]	Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger, “Densely Connected Convolutional Networks ”
[4]	Jayant, Caroline, vikiminki “Multi-label Classification of Movies” 
[5]	Elad Rapaport, “MovieLens-1M Deep Dive”
[6]	Prateek Joshi, “Predicting Movie Genres using NLP – An Awesome Introduction to Multi-Label Classification” 
[7]	Pulkit Sharma, “Build your First Multi-Label Image Classification Model in Python” 

