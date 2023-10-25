.. _cn_api_paddle_masked_scatter:

masked_scatter
-------------------------------

.. py:function:: paddle.masked_scatter(x, mask, value)

对于一个`目标Tensor`，根据`mask`信息，将`源Tensor`中的值按照顺序填充到`目标Tensor`中`mask`对应为`True`的位置。

参数
:::::::::
    - **x** (Tensor) - 输入的张量，支持的数据类型为float16、float32、float64、int32、int64、bool
    - **mask** (Tensor) - 用于指定填充位置的布尔值掩码张量，与 input 张量形状相同，或者可以广播成input张量的形状。
    - **value** (Tensor) - 待填充的张量，支持的数据类型为float16、float32、float64、int32、int64、bool，其中元素的数量应该不少于mask中True的个数，且元素数据类型要跟x中元素数据类型保持一致。

返回
:::::::::
多维 Tensor，数据类型与 ``x`` 相同，维度为广播后的形状。


代码示例
:::::::::
    .. code-block:: python
            
            >>> import paddle
            >>> x = paddle.randn([3,4])
            >>> mask = paddle.to_tensor([1.,0.5,1.,0.5]) > 0.5
            >>> value = paddle.ones([2,4], dtype="float32")
            >>> result = masked_scatter(x, mask, value)
            >>> print(result)
            Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            [[ 1.        , -2.59757781,  1.        , -2.37750435],
             [ 1.        , -0.11681330,  1.        ,  0.56991023],
             [ 1.        ,  2.51356053,  1.        ,  0.67361248]]) 
