#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 320

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;
/*****************************************************************************
 * 函 数 名  : svm_node
 * 负 责 人  : an
 * 创建日期  : 2017年5月5日
 * 函数功能  : 描述单一向量一个特征的结构体
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : 
 * 调用关系  : 
 * 其    它  : 

*****************************************************************************/
struct svm_node
{
	int index;	//特征的标号
	double value;//对应特征的值
};

//存储参与运算的所有数据集，及其所属类别
struct svm_problem
{
	int l,n;	//l:样本总数。n:查询总数
	int *query;	//查询的数组
	double *y;	//指向样本所属类别的数组
	struct svm_node **x;	//存储节点特征的数组
};

enum { L2R_RANK };	/* svm_type */
enum { LINEAR, POLY, RBF, PRECOMPUTED }; /* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
};

//
// svm_model
// 保存训练后的模型
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* 类别数number of classes, = 2 in regression/one class svm */
	int l;			/* 支持向量数total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* 判决函数中的系数coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* 判决函数中的常数constants in decision functions (rho[k*(k-1)/2]) */
	int *sv_indices;        /*支持向量的预测结果 sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void rank_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);

void svm_set_print_string_function(void (*print_func)(const char *));
void eval_list(double *label, double *target, int *query, int l, double *result_ret);

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
