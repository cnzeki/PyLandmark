#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "Landmark_clnf.h"
#include "PoseEstimator.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
using namespace boost::python;

static Landmark_clnf _instance;

int PyLandmark_init(const char* modelDir)
{
	return _instance.init(modelDir);
}

void PyLandmark_destroy()
{
	_instance.destroy();
}

template <typename Container>
boost::python::list stl2py(const Container& vec) {
  typedef typename Container::value_type T;
  boost::python::list lst;
  std::for_each(vec.begin(), vec.end(), [&](const T& t) { lst.append(t); });
  return lst;
}

template <typename Container>
void py2stl(const boost::python::list& lst, Container& vec) {
  typedef typename Container::value_type T;
  boost::python::stl_input_iterator<T> beg(lst), end;
  std::for_each(beg, end, [&](const T& t) { vec.push_back(t); });
}

boost::python::list PyLandmark_detect( cv::Mat image,
									boost::python::list _rect,
									boost::python::list _landmark5,
									int detectorType )
{
	// unpack to rect
    std::vector<int> rect;
    py2stl(_rect, rect);
    std::vector<int> landmark5;
    py2stl(_landmark5, landmark5);
	
	cv::Rect r(rect[0], rect[1], rect[2], rect[3]);
	// unpack to points
	std::vector<cv::Point> pts;
	for (size_t i = 0; i < landmark5.size(); i+=2)
	{
		cv::Point pt(landmark5[i], landmark5[i + 1]);
		pts.push_back(pt);
	}
	std::vector<cv::Point> landmark;
	_instance.landmark(image.data, image.cols, image.rows, r, pts, landmark, detectorType);

	// pack to int vec
	boost::python::list ret;
	for (size_t i = 0; i < landmark.size(); i++)
	{
		ret.append(landmark[i].x);
		ret.append(landmark[i].y);
	}
	return ret;
}


boost::python::list PyLandmark_getPose(boost::python::list _landmark)
{
	// unpack 
    std::vector<int> landmark;
    py2stl(_landmark, landmark);
	// unpack to points
	std::vector<cv::Point> pts;
	for (size_t i = 0; i < landmark.size(); i+=2)
	{
		cv::Point pt(landmark[i], landmark[i + 1]);
		pts.push_back(pt);
	}
	std::vector<float> eav;
    estimateEav(pts, eav);

	// pack to int vec
	boost::python::list ret = stl2py(eav);
	return ret;
}

#if (PY_VERSION_HEX >= 0x03000000)
    static void *init_ar() {
#else
    static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }
        
        
BOOST_PYTHON_MODULE(PyLandmark)
{
	using namespace boost::python;
    init_ar();

    //initialize converters
    to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();
        
	def("create", PyLandmark_init);
	def("destroy", PyLandmark_destroy);
	def("detect", PyLandmark_detect);
    def("getPose", PyLandmark_getPose);
    
	//class_<std::vector<int> >("IntVec")
	//	.def(vector_indexing_suite<std::vector<int> >())
	//	;

	//
	//class_<std::string>("stdstring", init<const char*>());
}
