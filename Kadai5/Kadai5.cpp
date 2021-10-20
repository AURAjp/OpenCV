#include <opencv2/opencv.hpp>

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_legacy2413d.lib")
#pragma comment(lib, "opencv_calib3d2413d.lib")
#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_legacy2413.lib")
#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_objdetect2413.lib")
#pragma comment(lib, "opencv_features2d2413.lib")
#pragma comment(lib, "opencv_calib3d2413.lib")
#endif

namespace cv
{
#if _MSC_VER >= 1200
#pragma warning( disable: 4244 )
#endif

    typedef ushort HT;

    typedef struct
    {
        HT coarse[16];
        HT fine[16][16];
    } Histogram;


#if CV_SSE2
#define MEDIAN_HAVE_SIMD 1

    static inline void histogram_add_simd(const HT x[16], HT y[16])
    {
        const __m128i* rx = (const __m128i*)x;
        __m128i* ry = (__m128i*)y;
        __m128i r0 = _mm_add_epi16(_mm_load_si128(ry + 0), _mm_load_si128(rx + 0));
        __m128i r1 = _mm_add_epi16(_mm_load_si128(ry + 1), _mm_load_si128(rx + 1));
        _mm_store_si128(ry + 0, r0);
        _mm_store_si128(ry + 1, r1);
    }

    static inline void histogram_sub_simd(const HT x[16], HT y[16])
    {
        const __m128i* rx = (const __m128i*)x;
        __m128i* ry = (__m128i*)y;
        __m128i r0 = _mm_sub_epi16(_mm_load_si128(ry + 0), _mm_load_si128(rx + 0));
        __m128i r1 = _mm_sub_epi16(_mm_load_si128(ry + 1), _mm_load_si128(rx + 1));
        _mm_store_si128(ry + 0, r0);
        _mm_store_si128(ry + 1, r1);
    }

#else
#define MEDIAN_HAVE_SIMD 0
#endif


    static inline void histogram_add(const HT x[16], HT y[16])
    {
        int i;
        for (i = 0; i < 16; ++i)
            y[i] = (HT)(y[i] + x[i]);
    }

    static inline void histogram_sub(const HT x[16], HT y[16])
    {
        int i;
        for (i = 0; i < 16; ++i)
            y[i] = (HT)(y[i] - x[i]);
    }

    static inline void histogram_muladd(int a, const HT x[16],
        HT y[16])
    {
        for (int i = 0; i < 16; ++i)
            y[i] = (HT)(y[i] + a * x[i]);
    }

    static void
        medianBlur_8u_O1(const Mat& _src, Mat& _dst, int ksize)
    {
        /**
         * HOP is short for Histogram OPeration. This macro makes an operation \a op on
         * histogram \a h for pixel value \a x. It takes care of handling both levels.
         */
#define HOP(h,x,op) \
    h.coarse[x>>4] op, \
    *((HT*)h.fine + x) op

#define COP(c,j,x,op) \
    h_coarse[ 16*(n*c+j) + (x>>4) ] op, \
    h_fine[ 16 * (n*(16*c+(x>>4)) + j) + (x & 0xF) ] op

        int cn = _dst.channels(), m = _dst.rows, r = (ksize - 1) / 2;
        size_t sstep = _src.step, dstep = _dst.step;
        Histogram H[4];
        HT luc[4][16];

        int STRIPE_SIZE = std::min(_dst.cols, 512 / cn);

        vector<HT> _h_coarse(1 * 16 * (STRIPE_SIZE + 2 * r) * cn + 16);
        vector<HT> _h_fine(16 * 16 * (STRIPE_SIZE + 2 * r) * cn + 16);
        HT* h_coarse = alignPtr(&_h_coarse[0], 16);
        HT* h_fine = alignPtr(&_h_fine[0], 16);
#if MEDIAN_HAVE_SIMD
        volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

        for (int x = 0; x < _dst.cols; x += STRIPE_SIZE)
        {
            int i, j, k, c, n = std::min(_dst.cols - x, STRIPE_SIZE) + r * 2;
            const uchar* src = _src.data + x * cn;
            uchar* dst = _dst.data + (x - r) * cn;

            memset(h_coarse, 0, 16 * n * cn * sizeof(h_coarse[0]));
            memset(h_fine, 0, 16 * 16 * n * cn * sizeof(h_fine[0]));

            // First row initialization
            for (c = 0; c < cn; c++)
            {
                for (j = 0; j < n; j++)
                    COP(c, j, src[cn * j + c], += r + 2);

                for (i = 1; i < r; i++)
                {
                    const uchar* p = src + sstep * std::min(i, m - 1);
                    for (j = 0; j < n; j++)
                        COP(c, j, p[cn * j + c], ++);
                }
            }

            for (i = 0; i < m; i++)
            {
                const uchar* p0 = src + sstep * std::max(0, i - r - 1);
                const uchar* p1 = src + sstep * std::min(m - 1, i + r);

                memset(H, 0, cn * sizeof(H[0]));
                memset(luc, 0, cn * sizeof(luc[0]));
                for (c = 0; c < cn; c++)
                {
                    // Update column histograms for the entire row.
                    for (j = 0; j < n; j++)
                    {
                        COP(c, j, p0[j * cn + c], --);
                        COP(c, j, p1[j * cn + c], ++);
                    }

                    // First column initialization
                    for (k = 0; k < 16; ++k)
                        histogram_muladd(2 * r + 1, &h_fine[16 * n * (16 * c + k)], &H[c].fine[k][0]);

#if MEDIAN_HAVE_SIMD
                    if (useSIMD)
                    {
                        for (j = 0; j < 2 * r; ++j)
                            histogram_add_simd(&h_coarse[16 * (n * c + j)], H[c].coarse);

                        for (j = r; j < n - r; j++)
                        {
                            int t = 2 * r * r + 2 * r, b, sum = 0;
                            HT* segment;

                            histogram_add_simd(&h_coarse[16 * (n * c + std::min(j + r, n - 1))], H[c].coarse);

                            // Find median at coarse level
                            for (k = 0; k < 16; ++k)
                            {
                                sum += H[c].coarse[k];
                                if (sum > t)
                                {
                                    sum -= H[c].coarse[k];
                                    break;
                                }
                            }
                            assert(k < 16);

                            /* Update corresponding histogram segment */
                            if (luc[c][k] <= j - r)
                            {
                                memset(&H[c].fine[k], 0, 16 * sizeof(HT));
                                for (luc[c][k] = j - r; luc[c][k] < MIN(j + r + 1, n); ++luc[c][k])
                                    histogram_add_simd(&h_fine[16 * (n * (16 * c + k) + luc[c][k])], H[c].fine[k]);

                                if (luc[c][k] < j + r + 1)
                                {
                                    histogram_muladd(j + r + 1 - n, &h_fine[16 * (n * (16 * c + k) + (n - 1))], &H[c].fine[k][0]);
                                    luc[c][k] = (HT)(j + r + 1);
                                }
                            }
                            else
                            {
                                for (; luc[c][k] < j + r + 1; ++luc[c][k])
                                {
                                    histogram_sub_simd(&h_fine[16 * (n * (16 * c + k) + MAX(luc[c][k] - 2 * r - 1, 0))], H[c].fine[k]);
                                    histogram_add_simd(&h_fine[16 * (n * (16 * c + k) + MIN(luc[c][k], n - 1))], H[c].fine[k]);
                                }
                            }

                            histogram_sub_simd(&h_coarse[16 * (n * c + MAX(j - r, 0))], H[c].coarse);

                            /* Find median in segment */
                            segment = H[c].fine[k];
                            for (b = 0; b < 16; b++)
                            {
                                sum += segment[b];
                                if (sum > t)
                                {
                                    dst[dstep * i + cn * j + c] = (uchar)(16 * k + b);
                                    break;
                                }
                            }
                            assert(b < 16);
                        }
                    }
                    else
#endif
                    {
                        for (j = 0; j < 2 * r; ++j)
                            histogram_add(&h_coarse[16 * (n * c + j)], H[c].coarse);

                        for (j = r; j < n - r; j++)
                        {
                            int t = 2 * r * r + 2 * r, b, sum = 0;
                            HT* segment;

                            histogram_add(&h_coarse[16 * (n * c + std::min(j + r, n - 1))], H[c].coarse);

                            // Find median at coarse level
                            for (k = 0; k < 16; ++k)
                            {
                                sum += H[c].coarse[k];
                                if (sum > t)
                                {
                                    sum -= H[c].coarse[k];
                                    break;
                                }
                            }
                            assert(k < 16);

                            /* Update corresponding histogram segment */
                            if (luc[c][k] <= j - r)
                            {
                                memset(&H[c].fine[k], 0, 16 * sizeof(HT));
                                for (luc[c][k] = j - r; luc[c][k] < MIN(j + r + 1, n); ++luc[c][k])
                                    histogram_add(&h_fine[16 * (n * (16 * c + k) + luc[c][k])], H[c].fine[k]);

                                if (luc[c][k] < j + r + 1)
                                {
                                    histogram_muladd(j + r + 1 - n, &h_fine[16 * (n * (16 * c + k) + (n - 1))], &H[c].fine[k][0]);
                                    luc[c][k] = (HT)(j + r + 1);
                                }
                            }
                            else
                            {
                                for (; luc[c][k] < j + r + 1; ++luc[c][k])
                                {
                                    histogram_sub(&h_fine[16 * (n * (16 * c + k) + MAX(luc[c][k] - 2 * r - 1, 0))], H[c].fine[k]);
                                    histogram_add(&h_fine[16 * (n * (16 * c + k) + MIN(luc[c][k], n - 1))], H[c].fine[k]);
                                }
                            }

                            histogram_sub(&h_coarse[16 * (n * c + MAX(j - r, 0))], H[c].coarse);

                            /* Find median in segment */
                            segment = H[c].fine[k];
                            for (b = 0; b < 16; b++)
                            {
                                sum += segment[b];
                                if (sum > t)
                                {
                                    dst[dstep * i + cn * j + c] = (uchar)(16 * k + b);
                                    break;
                                }
                            }
                            assert(b < 16);
                        }
                    }
                }
            }
        }

#undef HOP
#undef COP
    }


#if _MSC_VER >= 1200
#pragma warning( default: 4244 )
#endif

    static void
        medianBlur_8u_Om(const Mat& _src, Mat& _dst, int m)
    {
#define N  16
        int     zone0[4][N];
        int     zone1[4][N * N];
        int     x, y;
        int     n2 = m * m / 2;
        Size    size = _dst.size();
        const uchar* src = _src.data;
        uchar* dst = _dst.data;
        int     src_step = (int)_src.step, dst_step = (int)_dst.step;
        int     cn = _src.channels();
        const uchar* src_max = src + size.height * src_step;

#define UPDATE_ACC01( pix, cn, op ) \
    {                                   \
        int p = (pix);                  \
        zone1[cn][p] op;                \
        zone0[cn][p >> 4] op;           \
    }

        //CV_Assert( size.height >= nx && size.width >= nx );
        for (x = 0; x < size.width; x++, src += cn, dst += cn)
        {
            uchar* dst_cur = dst;
            const uchar* src_top = src;
            const uchar* src_bottom = src;
            int k, c;
            int src_step1 = src_step, dst_step1 = dst_step;

            if (x % 2 != 0)
            {
                src_bottom = src_top += src_step * (size.height - 1);
                dst_cur += dst_step * (size.height - 1);
                src_step1 = -src_step1;
                dst_step1 = -dst_step1;
            }

            // init accumulator
            memset(zone0, 0, sizeof(zone0[0]) * cn);
            memset(zone1, 0, sizeof(zone1[0]) * cn);

            for (y = 0; y <= m / 2; y++)
            {
                for (c = 0; c < cn; c++)
                {
                    if (y > 0)
                    {
                        for (k = 0; k < m * cn; k += cn)
                            UPDATE_ACC01(src_bottom[k + c], c, ++);
                    }
                    else
                    {
                        for (k = 0; k < m * cn; k += cn)
                            UPDATE_ACC01(src_bottom[k + c], c, += m / 2 + 1);
                    }
                }

                if ((src_step1 > 0 && y < size.height - 1) ||
                    (src_step1 < 0 && size.height - y - 1 > 0))
                    src_bottom += src_step1;
            }

            for (y = 0; y < size.height; y++, dst_cur += dst_step1)
            {
                // find median
                for (c = 0; c < cn; c++)
                {
                    int s = 0;
                    for (k = 0; ; k++)
                    {
                        int t = s + zone0[c][k];
                        if (t > n2) break;
                        s = t;
                    }

                    for (k *= N; ; k++)
                    {
                        s += zone1[c][k];
                        if (s > n2) break;
                    }

                    dst_cur[c] = (uchar)k;
                }

                if (y + 1 == size.height)
                    break;

                if (cn == 1)
                {
                    for (k = 0; k < m; k++)
                    {
                        int p = src_top[k];
                        int q = src_bottom[k];
                        zone1[0][p]--;
                        zone0[0][p >> 4]--;
                        zone1[0][q]++;
                        zone0[0][q >> 4]++;
                    }
                }
                else if (cn == 3)
                {
                    for (k = 0; k < m * 3; k += 3)
                    {
                        UPDATE_ACC01(src_top[k], 0, --);
                        UPDATE_ACC01(src_top[k + 1], 1, --);
                        UPDATE_ACC01(src_top[k + 2], 2, --);

                        UPDATE_ACC01(src_bottom[k], 0, ++);
                        UPDATE_ACC01(src_bottom[k + 1], 1, ++);
                        UPDATE_ACC01(src_bottom[k + 2], 2, ++);
                    }
                }
                else
                {
                    assert(cn == 4);
                    for (k = 0; k < m * 4; k += 4)
                    {
                        UPDATE_ACC01(src_top[k], 0, --);
                        UPDATE_ACC01(src_top[k + 1], 1, --);
                        UPDATE_ACC01(src_top[k + 2], 2, --);
                        UPDATE_ACC01(src_top[k + 3], 3, --);

                        UPDATE_ACC01(src_bottom[k], 0, ++);
                        UPDATE_ACC01(src_bottom[k + 1], 1, ++);
                        UPDATE_ACC01(src_bottom[k + 2], 2, ++);
                        UPDATE_ACC01(src_bottom[k + 3], 3, ++);
                    }
                }

                if ((src_step1 > 0 && src_bottom + src_step1 < src_max) ||
                    (src_step1 < 0 && src_bottom + src_step1 >= src))
                    src_bottom += src_step1;

                if (y >= m / 2)
                    src_top += src_step1;
            }
        }
#undef N
#undef UPDATE_ACC
    }
   
    struct MinMax8u
    {
        typedef uchar value_type;
        typedef int arg_type;
        enum { SIZE = 1 };
        arg_type load(const uchar* ptr) { return *ptr; }
        void store(uchar* ptr, arg_type val) { *ptr = (uchar)val; }
        void operator()(arg_type& a, arg_type& b) const
        {
            int t = a - b + 256;
            b += t; a -= t;
        }
    };

    struct MinMax16u
    {
        typedef ushort value_type;
        typedef int arg_type;
        enum { SIZE = 1 };
        arg_type load(const ushort* ptr) { return *ptr; }
        void store(ushort* ptr, arg_type val) { *ptr = (ushort)val; }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = std::min(a, b);
            b = std::max(b, t);
        }
    };

    struct MinMax16s
    {
        typedef short value_type;
        typedef int arg_type;
        enum { SIZE = 1 };
        arg_type load(const short* ptr) { return *ptr; }
        void store(short* ptr, arg_type val) { *ptr = (short)val; }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = std::min(a, b);
            b = std::max(b, t);
        }
    };

    struct MinMax32f
    {
        typedef float value_type;
        typedef float arg_type;
        enum { SIZE = 1 };
        arg_type load(const float* ptr) { return *ptr; }
        void store(float* ptr, arg_type val) { *ptr = val; }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = std::min(a, b);
            b = std::max(b, t);
        }
    };

#if CV_SSE2

    struct MinMaxVec8u
    {
        typedef uchar value_type;
        typedef __m128i arg_type;
        enum { SIZE = 16 };
        arg_type load(const uchar* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
        void store(uchar* ptr, arg_type val) { _mm_storeu_si128((__m128i*)ptr, val); }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = _mm_min_epu8(a, b);
            b = _mm_max_epu8(b, t);
        }
    };


    struct MinMaxVec16u
    {
        typedef ushort value_type;
        typedef __m128i arg_type;
        enum { SIZE = 8 };
        arg_type load(const ushort* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
        void store(ushort* ptr, arg_type val) { _mm_storeu_si128((__m128i*)ptr, val); }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = _mm_subs_epu16(a, b);
            a = _mm_subs_epu16(a, t);
            b = _mm_adds_epu16(b, t);
        }
    };


    struct MinMaxVec16s
    {
        typedef short value_type;
        typedef __m128i arg_type;
        enum { SIZE = 8 };
        arg_type load(const short* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
        void store(short* ptr, arg_type val) { _mm_storeu_si128((__m128i*)ptr, val); }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = _mm_min_epi16(a, b);
            b = _mm_max_epi16(b, t);
        }
    };


    struct MinMaxVec32f
    {
        typedef float value_type;
        typedef __m128 arg_type;
        enum { SIZE = 4 };
        arg_type load(const float* ptr) { return _mm_loadu_ps(ptr); }
        void store(float* ptr, arg_type val) { _mm_storeu_ps(ptr, val); }
        void operator()(arg_type& a, arg_type& b) const
        {
            arg_type t = a;
            a = _mm_min_ps(a, b);
            b = _mm_max_ps(b, t);
        }
    };


#else

    typedef MinMax8u MinMaxVec8u;
    typedef MinMax16u MinMaxVec16u;
    typedef MinMax16s MinMaxVec16s;
    typedef MinMax32f MinMaxVec32f;

#endif

    template<class Op, class VecOp>
    static void
        medianBlur_SortNet(const Mat& _src, Mat& _dst, int m)
    {
        typedef typename Op::value_type T;
        typedef typename Op::arg_type WT;
        typedef typename VecOp::arg_type VT;

        const T* src = (const T*)_src.data;
        T* dst = (T*)_dst.data;
        int sstep = (int)(_src.step / sizeof(T));
        int dstep = (int)(_dst.step / sizeof(T));
        Size size = _dst.size();
        int i, j, k, cn = _src.channels();
        Op op;
        VecOp vop;
        volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);

        if (m == 3)
        {
            if (size.width == 1 || size.height == 1)
            {
                int len = size.width + size.height - 1;
                int sdelta = size.height == 1 ? cn : sstep;
                int sdelta0 = size.height == 1 ? 0 : sstep - cn;
                int ddelta = size.height == 1 ? cn : dstep;

                for (i = 0; i < len; i++, src += sdelta0, dst += ddelta)
                    for (j = 0; j < cn; j++, src++)
                    {
                        WT p0 = src[i > 0 ? -sdelta : 0];
                        WT p1 = src[0];
                        WT p2 = src[i < len - 1 ? sdelta : 0];

                        op(p0, p1); op(p1, p2); op(p0, p1);
                        dst[j] = (T)p1;
                    }
                return;
            }

            size.width *= cn;
            for (i = 0; i < size.height; i++, dst += dstep)
            {
                const T* row0 = src + std::max(i - 1, 0) * sstep;
                const T* row1 = src + i * sstep;
                const T* row2 = src + std::min(i + 1, size.height - 1) * sstep;
                int limit = useSIMD ? cn : size.width;

                for (j = 0;; )
                {
                    for (; j < limit; j++)
                    {
                        int j0 = j >= cn ? j - cn : j;
                        int j2 = j < size.width - cn ? j + cn : j;
                        WT p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
                        WT p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
                        WT p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

                        op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
                        op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
                        op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
                        op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
                        op(p4, p2); op(p6, p4); op(p4, p2);
                        dst[j] = (T)p4;
                    }

                    if (limit == size.width)
                        break;

                    for (; j <= size.width - VecOp::SIZE - cn; j += VecOp::SIZE)
                    {
                        VT p0 = vop.load(row0 + j - cn), p1 = vop.load(row0 + j), p2 = vop.load(row0 + j + cn);
                        VT p3 = vop.load(row1 + j - cn), p4 = vop.load(row1 + j), p5 = vop.load(row1 + j + cn);
                        VT p6 = vop.load(row2 + j - cn), p7 = vop.load(row2 + j), p8 = vop.load(row2 + j + cn);

                        vop(p1, p2); vop(p4, p5); vop(p7, p8); vop(p0, p1);
                        vop(p3, p4); vop(p6, p7); vop(p1, p2); vop(p4, p5);
                        vop(p7, p8); vop(p0, p3); vop(p5, p8); vop(p4, p7);
                        vop(p3, p6); vop(p1, p4); vop(p2, p5); vop(p4, p7);
                        vop(p4, p2); vop(p6, p4); vop(p4, p2);
                        vop.store(dst + j, p4);
                    }

                    limit = size.width;
                }
            }
        }
        else if (m == 5)
        {
            if (size.width == 1 || size.height == 1)
            {
                int len = size.width + size.height - 1;
                int sdelta = size.height == 1 ? cn : sstep;
                int sdelta0 = size.height == 1 ? 0 : sstep - cn;
                int ddelta = size.height == 1 ? cn : dstep;

                for (i = 0; i < len; i++, src += sdelta0, dst += ddelta)
                    for (j = 0; j < cn; j++, src++)
                    {
                        int i1 = i > 0 ? -sdelta : 0;
                        int i0 = i > 1 ? -sdelta * 2 : i1;
                        int i3 = i < len - 1 ? sdelta : 0;
                        int i4 = i < len - 2 ? sdelta * 2 : i3;
                        WT p0 = src[i0], p1 = src[i1], p2 = src[0], p3 = src[i3], p4 = src[i4];

                        op(p0, p1); op(p3, p4); op(p2, p3); op(p3, p4); op(p0, p2);
                        op(p2, p4); op(p1, p3); op(p1, p2);
                        dst[j] = (T)p2;
                    }
                return;
            }

            size.width *= cn;
            for (i = 0; i < size.height; i++, dst += dstep)
            {
                const T* row[5];
                row[0] = src + std::max(i - 2, 0) * sstep;
                row[1] = src + std::max(i - 1, 0) * sstep;
                row[2] = src + i * sstep;
                row[3] = src + std::min(i + 1, size.height - 1) * sstep;
                row[4] = src + std::min(i + 2, size.height - 1) * sstep;
                int limit = useSIMD ? cn * 2 : size.width;

                for (j = 0;; )
                {
                    for (; j < limit; j++)
                    {
                        WT p[25];
                        int j1 = j >= cn ? j - cn : j;
                        int j0 = j >= cn * 2 ? j - cn * 2 : j1;
                        int j3 = j < size.width - cn ? j + cn : j;
                        int j4 = j < size.width - cn * 2 ? j + cn * 2 : j3;
                        for (k = 0; k < 5; k++)
                        {
                            const T* rowk = row[k];
                            p[k * 5] = rowk[j0]; p[k * 5 + 1] = rowk[j1];
                            p[k * 5 + 2] = rowk[j]; p[k * 5 + 3] = rowk[j3];
                            p[k * 5 + 4] = rowk[j4];
                        }

                        op(p[1], p[2]); op(p[0], p[1]); op(p[1], p[2]); op(p[4], p[5]); op(p[3], p[4]);
                        op(p[4], p[5]); op(p[0], p[3]); op(p[2], p[5]); op(p[2], p[3]); op(p[1], p[4]);
                        op(p[1], p[2]); op(p[3], p[4]); op(p[7], p[8]); op(p[6], p[7]); op(p[7], p[8]);
                        op(p[10], p[11]); op(p[9], p[10]); op(p[10], p[11]); op(p[6], p[9]); op(p[8], p[11]);
                        op(p[8], p[9]); op(p[7], p[10]); op(p[7], p[8]); op(p[9], p[10]); op(p[0], p[6]);
                        op(p[4], p[10]); op(p[4], p[6]); op(p[2], p[8]); op(p[2], p[4]); op(p[6], p[8]);
                        op(p[1], p[7]); op(p[5], p[11]); op(p[5], p[7]); op(p[3], p[9]); op(p[3], p[5]);
                        op(p[7], p[9]); op(p[1], p[2]); op(p[3], p[4]); op(p[5], p[6]); op(p[7], p[8]);
                        op(p[9], p[10]); op(p[13], p[14]); op(p[12], p[13]); op(p[13], p[14]); op(p[16], p[17]);
                        op(p[15], p[16]); op(p[16], p[17]); op(p[12], p[15]); op(p[14], p[17]); op(p[14], p[15]);
                        op(p[13], p[16]); op(p[13], p[14]); op(p[15], p[16]); op(p[19], p[20]); op(p[18], p[19]);
                        op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[21], p[23]); op(p[22], p[24]);
                        op(p[22], p[23]); op(p[18], p[21]); op(p[20], p[23]); op(p[20], p[21]); op(p[19], p[22]);
                        op(p[22], p[24]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[12], p[18]);
                        op(p[16], p[22]); op(p[16], p[18]); op(p[14], p[20]); op(p[20], p[24]); op(p[14], p[16]);
                        op(p[18], p[20]); op(p[22], p[24]); op(p[13], p[19]); op(p[17], p[23]); op(p[17], p[19]);
                        op(p[15], p[21]); op(p[15], p[17]); op(p[19], p[21]); op(p[13], p[14]); op(p[15], p[16]);
                        op(p[17], p[18]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[0], p[12]);
                        op(p[8], p[20]); op(p[8], p[12]); op(p[4], p[16]); op(p[16], p[24]); op(p[12], p[16]);
                        op(p[2], p[14]); op(p[10], p[22]); op(p[10], p[14]); op(p[6], p[18]); op(p[6], p[10]);
                        op(p[10], p[12]); op(p[1], p[13]); op(p[9], p[21]); op(p[9], p[13]); op(p[5], p[17]);
                        op(p[13], p[17]); op(p[3], p[15]); op(p[11], p[23]); op(p[11], p[15]); op(p[7], p[19]);
                        op(p[7], p[11]); op(p[11], p[13]); op(p[11], p[12]);
                        dst[j] = (T)p[12];
                    }

                    if (limit == size.width)
                        break;

                    for (; j <= size.width - VecOp::SIZE - cn * 2; j += VecOp::SIZE)
                    {
                        VT p[25];
                        for (k = 0; k < 5; k++)
                        {
                            const T* rowk = row[k];
                            p[k * 5] = vop.load(rowk + j - cn * 2); p[k * 5 + 1] = vop.load(rowk + j - cn);
                            p[k * 5 + 2] = vop.load(rowk + j); p[k * 5 + 3] = vop.load(rowk + j + cn);
                            p[k * 5 + 4] = vop.load(rowk + j + cn * 2);
                        }

                        vop(p[1], p[2]); vop(p[0], p[1]); vop(p[1], p[2]); vop(p[4], p[5]); vop(p[3], p[4]);
                        vop(p[4], p[5]); vop(p[0], p[3]); vop(p[2], p[5]); vop(p[2], p[3]); vop(p[1], p[4]);
                        vop(p[1], p[2]); vop(p[3], p[4]); vop(p[7], p[8]); vop(p[6], p[7]); vop(p[7], p[8]);
                        vop(p[10], p[11]); vop(p[9], p[10]); vop(p[10], p[11]); vop(p[6], p[9]); vop(p[8], p[11]);
                        vop(p[8], p[9]); vop(p[7], p[10]); vop(p[7], p[8]); vop(p[9], p[10]); vop(p[0], p[6]);
                        vop(p[4], p[10]); vop(p[4], p[6]); vop(p[2], p[8]); vop(p[2], p[4]); vop(p[6], p[8]);
                        vop(p[1], p[7]); vop(p[5], p[11]); vop(p[5], p[7]); vop(p[3], p[9]); vop(p[3], p[5]);
                        vop(p[7], p[9]); vop(p[1], p[2]); vop(p[3], p[4]); vop(p[5], p[6]); vop(p[7], p[8]);
                        vop(p[9], p[10]); vop(p[13], p[14]); vop(p[12], p[13]); vop(p[13], p[14]); vop(p[16], p[17]);
                        vop(p[15], p[16]); vop(p[16], p[17]); vop(p[12], p[15]); vop(p[14], p[17]); vop(p[14], p[15]);
                        vop(p[13], p[16]); vop(p[13], p[14]); vop(p[15], p[16]); vop(p[19], p[20]); vop(p[18], p[19]);
                        vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[21], p[23]); vop(p[22], p[24]);
                        vop(p[22], p[23]); vop(p[18], p[21]); vop(p[20], p[23]); vop(p[20], p[21]); vop(p[19], p[22]);
                        vop(p[22], p[24]); vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[12], p[18]);
                        vop(p[16], p[22]); vop(p[16], p[18]); vop(p[14], p[20]); vop(p[20], p[24]); vop(p[14], p[16]);
                        vop(p[18], p[20]); vop(p[22], p[24]); vop(p[13], p[19]); vop(p[17], p[23]); vop(p[17], p[19]);
                        vop(p[15], p[21]); vop(p[15], p[17]); vop(p[19], p[21]); vop(p[13], p[14]); vop(p[15], p[16]);
                        vop(p[17], p[18]); vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[0], p[12]);
                        vop(p[8], p[20]); vop(p[8], p[12]); vop(p[4], p[16]); vop(p[16], p[24]); vop(p[12], p[16]);
                        vop(p[2], p[14]); vop(p[10], p[22]); vop(p[10], p[14]); vop(p[6], p[18]); vop(p[6], p[10]);
                        vop(p[10], p[12]); vop(p[1], p[13]); vop(p[9], p[21]); vop(p[9], p[13]); vop(p[5], p[17]);
                        vop(p[13], p[17]); vop(p[3], p[15]); vop(p[11], p[23]); vop(p[11], p[15]); vop(p[7], p[19]);
                        vop(p[7], p[11]); vop(p[11], p[13]); vop(p[11], p[12]);
                        vop.store(dst + j, p[12]);
                    }

                    limit = size.width;
                }
            }
        }
    }


    void my_medianBlur(const Mat& src0, Mat& dst, int ksize)
    {
        if (ksize <= 1)
        {
            src0.copyTo(dst);
            return;
        }

        CV_Assert(ksize % 2 == 1);

        Size size = src0.size();
        int cn = src0.channels();
        bool useSortNet = ksize == 3 || (ksize == 5
#if !CV_SSE2
            && src0.depth() > CV_8U
#endif
            );

        dst.create(src0.size(), src0.type());
        Mat src;
        if (useSortNet)
        {
            if (dst.data != src0.data)
                src = src0;
            else
                src0.copyTo(src);
        }
        else
            cv::copyMakeBorder(src0, src, 0, 0, ksize / 2, ksize / 2, BORDER_REPLICATE);

        if (useSortNet)
        {
            if (src.depth() == CV_8U)
                medianBlur_SortNet<MinMax8u, MinMaxVec8u>(src, dst, ksize);
            else if (src.depth() == CV_16U)
                medianBlur_SortNet<MinMax16u, MinMaxVec16u>(src, dst, ksize);
            else if (src.depth() == CV_16S)
                medianBlur_SortNet<MinMax16s, MinMaxVec16s>(src, dst, ksize);
            else if (src.depth() == CV_32F)
                medianBlur_SortNet<MinMax32f, MinMaxVec32f>(src, dst, ksize);
            else
                CV_Error(CV_StsUnsupportedFormat, "");
            return;
        }

        CV_Assert(src.depth() == CV_8U && (cn == 1 || cn == 3 || cn == 4));

        double img_size_mp = (double)(size.width * size.height) / (1 << 20);
        if (ksize <= 3 + (img_size_mp < 1 ? 12 : img_size_mp < 4 ? 6 : 2) * (MEDIAN_HAVE_SIMD && checkHardwareSupport(CV_CPU_SSE2) ? 1 : 3))
            medianBlur_8u_Om(src, dst, ksize);
        else
            medianBlur_8u_O1(src, dst, ksize);
    }
}

int main()
{
    // 画像の読み込み
    auto src_img = cv::imread("./img/in.jpg", cv::IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty())
    {
        return -1;
    }

    // 入力画像のアスペクト比
    auto aspect_ratio = (double)src_img.rows / src_img.cols;
    // 出力画像の横幅
    constexpr auto WIDTH = 800;
    // アスペクト比を保持した高さ
    int height = aspect_ratio * WIDTH;

    // リサイズ用画像領域の確保
    cv::Mat resize_img(height, WIDTH, src_img.type());
    // リサイズ
    resize(src_img, resize_img, resize_img.size());

    cv::Mat out5_img;
    my_medianBlur(resize_img, out5_img, 5);
    cv::Mat out11_img;
    my_medianBlur(resize_img, out11_img, 11);
    cv::Mat out21_img;
    my_medianBlur(resize_img, out21_img, 21);

    // 結果表示
    cv::imshow("out", out5_img);
    cv::imshow("out", out11_img);
    cv::imshow("out", out21_img);

    cv::imwrite("./img/out5.png", out5_img);
    cv::imwrite("./img/out11.png", out11_img);
    cv::imwrite("./img/out21.png", out21_img);

    // キーボードが押されるまでwait
    cv::waitKey(0);

    return 0;
}