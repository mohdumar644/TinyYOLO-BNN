import unittest
import torch
from lightnet.network.layer import *

# this array was generated based on a manually validated implementation
reorg_forward_expected_output = torch.FloatTensor([
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
    32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0,
    64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0,
    96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
    128.0, 130.0, 132.0, 134.0, 136.0, 138.0, 140.0, 142.0,
    160.0, 162.0, 164.0, 166.0, 168.0, 170.0, 172.0, 174.0,
    192.0, 194.0, 196.0, 198.0, 200.0, 202.0, 204.0, 206.0,
    224.0, 226.0, 228.0, 230.0, 232.0, 234.0, 236.0, 238.0,
    256.0, 258.0, 260.0, 262.0, 264.0, 266.0, 268.0, 270.0,
    288.0, 290.0, 292.0, 294.0, 296.0, 298.0, 300.0, 302.0,
    320.0, 322.0, 324.0, 326.0, 328.0, 330.0, 332.0, 334.0,
    352.0, 354.0, 356.0, 358.0, 360.0, 362.0, 364.0, 366.0,
    384.0, 386.0, 388.0, 390.0, 392.0, 394.0, 396.0, 398.0,
    416.0, 418.0, 420.0, 422.0, 424.0, 426.0, 428.0, 430.0,
    448.0, 450.0, 452.0, 454.0, 456.0, 458.0, 460.0, 462.0,
    480.0, 482.0, 484.0, 486.0, 488.0, 490.0, 492.0, 494.0,
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
    33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0,
    65.0, 67.0, 69.0, 71.0, 73.0, 75.0, 77.0, 79.0,
    97.0, 99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0,
    129.0, 131.0, 133.0, 135.0, 137.0, 139.0, 141.0, 143.0,
    161.0, 163.0, 165.0, 167.0, 169.0, 171.0, 173.0, 175.0,
    193.0, 195.0, 197.0, 199.0, 201.0, 203.0, 205.0, 207.0,
    225.0, 227.0, 229.0, 231.0, 233.0, 235.0, 237.0, 239.0,
    257.0, 259.0, 261.0, 263.0, 265.0, 267.0, 269.0, 271.0,
    289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 301.0, 303.0,
    321.0, 323.0, 325.0, 327.0, 329.0, 331.0, 333.0, 335.0,
    353.0, 355.0, 357.0, 359.0, 361.0, 363.0, 365.0, 367.0,
    385.0, 387.0, 389.0, 391.0, 393.0, 395.0, 397.0, 399.0,
    417.0, 419.0, 421.0, 423.0, 425.0, 427.0, 429.0, 431.0,
    449.0, 451.0, 453.0, 455.0, 457.0, 459.0, 461.0, 463.0,
    481.0, 483.0, 485.0, 487.0, 489.0, 491.0, 493.0, 495.0,
    16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0,
    80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
    112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0, 126.0,
    144.0, 146.0, 148.0, 150.0, 152.0, 154.0, 156.0, 158.0,
    176.0, 178.0, 180.0, 182.0, 184.0, 186.0, 188.0, 190.0,
    208.0, 210.0, 212.0, 214.0, 216.0, 218.0, 220.0, 222.0,
    240.0, 242.0, 244.0, 246.0, 248.0, 250.0, 252.0, 254.0,
    272.0, 274.0, 276.0, 278.0, 280.0, 282.0, 284.0, 286.0,
    304.0, 306.0, 308.0, 310.0, 312.0, 314.0, 316.0, 318.0,
    336.0, 338.0, 340.0, 342.0, 344.0, 346.0, 348.0, 350.0,
    368.0, 370.0, 372.0, 374.0, 376.0, 378.0, 380.0, 382.0,
    400.0, 402.0, 404.0, 406.0, 408.0, 410.0, 412.0, 414.0,
    432.0, 434.0, 436.0, 438.0, 440.0, 442.0, 444.0, 446.0,
    464.0, 466.0, 468.0, 470.0, 472.0, 474.0, 476.0, 478.0,
    496.0, 498.0, 500.0, 502.0, 504.0, 506.0, 508.0, 510.0,
    17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0,
    49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 63.0,
    81.0, 83.0, 85.0, 87.0, 89.0, 91.0, 93.0, 95.0,
    113.0, 115.0, 117.0, 119.0, 121.0, 123.0, 125.0, 127.0,
    145.0, 147.0, 149.0, 151.0, 153.0, 155.0, 157.0, 159.0,
    177.0, 179.0, 181.0, 183.0, 185.0, 187.0, 189.0, 191.0,
    209.0, 211.0, 213.0, 215.0, 217.0, 219.0, 221.0, 223.0,
    241.0, 243.0, 245.0, 247.0, 249.0, 251.0, 253.0, 255.0,
    273.0, 275.0, 277.0, 279.0, 281.0, 283.0, 285.0, 287.0,
    305.0, 307.0, 309.0, 311.0, 313.0, 315.0, 317.0, 319.0,
    337.0, 339.0, 341.0, 343.0, 345.0, 347.0, 349.0, 351.0,
    369.0, 371.0, 373.0, 375.0, 377.0, 379.0, 381.0, 383.0,
    401.0, 403.0, 405.0, 407.0, 409.0, 411.0, 413.0, 415.0,
    433.0, 435.0, 437.0, 439.0, 441.0, 443.0, 445.0, 447.0,
    465.0, 467.0, 469.0, 471.0, 473.0, 475.0, 477.0, 479.0,
    497.0, 499.0, 501.0, 503.0, 505.0, 507.0, 509.0, 511.0])


class TestYoloReorg(unittest.TestCase):
    def setUp(self):
        self.reorg = Reorg(2)

        # create and initialize input tensor
        self.input = torch.FloatTensor(1, 8, 8, 8)
        z = self.input.view(-1)
        for count in range(len(z)):
            z[count] = count

        # already create variable of it
        self.input = torch.autograd.Variable(self.input)

    def tearDown(self):
        pass

    def test_dimensions_forward_cpu(self):
        """Validate that the dimensions of the output tensor are
        correct given a tensor with known input dimensions.
        Test CPU implementation
        """
        output = self.reorg.forward(self.input)
        self.assertEqual(output.size(), torch.Size([1, 32, 4, 4]))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_dimensions_forward_cuda(self):
        """Validate that the dimensions of the output tensor are
        correct given a tensor with known input dimensions.
        Test CUDA implementation
        """
        self.input = self.input.cuda()
        output = self.reorg.forward(self.input)
        self.assertEqual(output.size(), torch.Size([1, 32, 4, 4]))

    def test_forward_cpu(self):
        """Validate that the reorg layer puts the input elements to
        the correct locations in the output tensor.
        Test CPU implementation
        """
        output = self.reorg.forward(self.input)
        equal_elements = torch.eq(output.data, reorg_forward_expected_output.view(1, 32, 4, 4))
        self.assertTrue(equal_elements.all())

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_forward_cuda(self):
        """Validate that the reorg layer puts the input elements to
        the correct locations in the output tensor.
        Test CUDA implementation
        """
        self.input = self.input.cuda()
        output = self.reorg.forward(self.input)
        equal_elements = torch.eq(output.data.cpu(), reorg_forward_expected_output.view(1, 32, 4, 4))
        self.assertTrue(equal_elements.all())


if __name__ == '__main__':
    unittest.main()
