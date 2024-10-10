
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem reward-15x15)
    (:domain reward-strips)

    (:objects
        c_0_0 c_0_1 c_0_10 c_0_11 c_0_12 c_0_13 c_0_14 c_0_2 c_0_3 c_0_4 c_0_5 c_0_6 c_0_7 c_0_8 c_0_9 c_10_0 c_10_1 c_10_10 c_10_11 c_10_12 c_10_13 c_10_14 c_10_2 c_10_3 c_10_4 c_10_5 c_10_6 c_10_7 c_10_8 c_10_9 c_11_0 c_11_1 c_11_10 c_11_11 c_11_12 c_11_13 c_11_14 c_11_2 c_11_3 c_11_4 c_11_5 c_11_6 c_11_7 c_11_8 c_11_9 c_12_0 c_12_1 c_12_10 c_12_11 c_12_12 c_12_13 c_12_14 c_12_2 c_12_3 c_12_4 c_12_5 c_12_6 c_12_7 c_12_8 c_12_9 c_13_0 c_13_1 c_13_10 c_13_11 c_13_12 c_13_13 c_13_14 c_13_2 c_13_3 c_13_4 c_13_5 c_13_6 c_13_7 c_13_8 c_13_9 c_14_0 c_14_1 c_14_10 c_14_11 c_14_12 c_14_13 c_14_14 c_14_2 c_14_3 c_14_4 c_14_5 c_14_6 c_14_7 c_14_8 c_14_9 c_1_0 c_1_1 c_1_10 c_1_11 c_1_12 c_1_13 c_1_14 c_1_2 c_1_3 c_1_4 c_1_5 c_1_6 c_1_7 c_1_8 c_1_9 c_2_0 c_2_1 c_2_10 c_2_11 c_2_12 c_2_13 c_2_14 c_2_2 c_2_3 c_2_4 c_2_5 c_2_6 c_2_7 c_2_8 c_2_9 c_3_0 c_3_1 c_3_10 c_3_11 c_3_12 c_3_13 c_3_14 c_3_2 c_3_3 c_3_4 c_3_5 c_3_6 c_3_7 c_3_8 c_3_9 c_4_0 c_4_1 c_4_10 c_4_11 c_4_12 c_4_13 c_4_14 c_4_2 c_4_3 c_4_4 c_4_5 c_4_6 c_4_7 c_4_8 c_4_9 c_5_0 c_5_1 c_5_10 c_5_11 c_5_12 c_5_13 c_5_14 c_5_2 c_5_3 c_5_4 c_5_5 c_5_6 c_5_7 c_5_8 c_5_9 c_6_0 c_6_1 c_6_10 c_6_11 c_6_12 c_6_13 c_6_14 c_6_2 c_6_3 c_6_4 c_6_5 c_6_6 c_6_7 c_6_8 c_6_9 c_7_0 c_7_1 c_7_10 c_7_11 c_7_12 c_7_13 c_7_14 c_7_2 c_7_3 c_7_4 c_7_5 c_7_6 c_7_7 c_7_8 c_7_9 c_8_0 c_8_1 c_8_10 c_8_11 c_8_12 c_8_13 c_8_14 c_8_2 c_8_3 c_8_4 c_8_5 c_8_6 c_8_7 c_8_8 c_8_9 c_9_0 c_9_1 c_9_10 c_9_11 c_9_12 c_9_13 c_9_14 c_9_2 c_9_3 c_9_4 c_9_5 c_9_6 c_9_7 c_9_8 c_9_9 - cell
    )

    (:init
        (adjacent c_9_5 c_8_5)
        (adjacent c_1_9 c_0_9)
        (adjacent c_4_6 c_3_6)
        (adjacent c_9_13 c_9_12)
        (adjacent c_8_6 c_8_5)
        (adjacent c_11_10 c_11_9)
        (adjacent c_5_9 c_6_9)
        (adjacent c_11_14 c_10_14)
        (adjacent c_14_0 c_14_1)
        (adjacent c_2_3 c_2_2)
        (adjacent c_3_0 c_4_0)
        (adjacent c_1_10 c_2_10)
        (adjacent c_13_3 c_14_3)
        (adjacent c_11_11 c_11_10)
        (adjacent c_11_10 c_12_10)
        (adjacent c_4_8 c_5_8)
        (adjacent c_12_6 c_12_7)
        (adjacent c_11_9 c_10_9)
        (adjacent c_3_0 c_2_0)
        (adjacent c_3_3 c_3_4)
        (adjacent c_11_12 c_10_12)
        (adjacent c_12_12 c_12_13)
        (adjacent c_6_2 c_6_3)
        (adjacent c_9_3 c_8_3)
        (adjacent c_3_3 c_2_3)
        (adjacent c_14_13 c_14_14)
        (adjacent c_9_1 c_8_1)
        (adjacent c_11_13 c_10_13)
        (adjacent c_6_9 c_6_10)
        (adjacent c_0_5 c_0_4)
        (adjacent c_7_0 c_8_0)
        (adjacent c_13_8 c_13_7)
        (adjacent c_6_12 c_5_12)
        (adjacent c_2_1 c_2_2)
        (adjacent c_0_12 c_0_11)
        (adjacent c_6_3 c_5_3)
        (adjacent c_4_8 c_4_9)
        (adjacent c_3_8 c_3_7)
        (adjacent c_6_12 c_6_13)
        (adjacent c_12_5 c_12_4)
        (adjacent c_13_4 c_13_5)
        (adjacent c_9_12 c_9_11)
        (adjacent c_8_5 c_8_6)
        (adjacent c_1_12 c_0_12)
        (adjacent c_10_6 c_11_6)
        (adjacent c_14_8 c_14_7)
        (adjacent c_3_1 c_3_0)
        (adjacent c_3_11 c_3_10)
        (adjacent c_7_7 c_7_6)
        (adjacent c_10_11 c_9_11)
        (adjacent c_13_13 c_14_13)
        (adjacent c_4_5 c_4_4)
        (adjacent c_5_9 c_5_10)
        (adjacent c_11_7 c_11_6)
        (adjacent c_2_2 c_1_2)
        (adjacent c_5_6 c_6_6)
        (adjacent c_9_9 c_8_9)
        (adjacent c_8_10 c_8_11)
        (adjacent c_13_9 c_14_9)
        (adjacent c_10_11 c_10_12)
        (adjacent c_2_8 c_1_8)
        (adjacent c_2_2 c_2_3)
        (adjacent c_6_2 c_7_2)
        (adjacent c_8_5 c_8_4)
        (adjacent c_9_6 c_8_6)
        (adjacent c_8_6 c_7_6)
        (adjacent c_9_6 c_9_7)
        (adjacent c_9_0 c_9_1)
        (adjacent c_7_9 c_8_9)
        (adjacent c_14_8 c_14_9)
        (adjacent c_6_7 c_6_6)
        (adjacent c_11_11 c_12_11)
        (adjacent c_2_9 c_2_8)
        (adjacent c_1_1 c_1_0)
        (adjacent c_12_3 c_11_3)
        (adjacent c_13_2 c_14_2)
        (adjacent c_4_5 c_5_5)
        (adjacent c_14_0 c_13_0)
        (adjacent c_5_7 c_4_7)
        (adjacent c_13_7 c_13_8)
        (adjacent c_6_13 c_7_13)
        (adjacent c_4_3 c_4_4)
        (adjacent c_4_12 c_3_12)
        (adjacent c_6_5 c_6_4)
        (adjacent c_4_1 c_4_0)
        (adjacent c_14_5 c_13_5)
        (adjacent c_6_8 c_5_8)
        (adjacent c_11_9 c_12_9)
        (adjacent c_3_1 c_2_1)
        (adjacent c_1_2 c_1_3)
        (adjacent c_6_1 c_6_0)
        (adjacent c_1_8 c_0_8)
        (adjacent c_11_0 c_10_0)
        (adjacent c_5_13 c_5_14)
        (adjacent c_4_11 c_3_11)
        (adjacent c_5_13 c_6_13)
        (adjacent c_7_7 c_6_7)
        (adjacent c_10_9 c_10_8)
        (adjacent c_12_9 c_12_8)
        (adjacent c_0_4 c_0_5)
        (adjacent c_6_11 c_7_11)
        (adjacent c_9_4 c_9_5)
        (adjacent c_2_0 c_2_1)
        (adjacent c_2_13 c_1_13)
        (adjacent c_5_14 c_4_14)
        (adjacent c_9_2 c_9_1)
        (adjacent c_13_12 c_12_12)
        (adjacent c_7_8 c_7_9)
        (adjacent c_14_6 c_14_7)
        (adjacent c_13_0 c_14_0)
        (adjacent c_11_14 c_11_13)
        (adjacent c_10_11 c_11_11)
        (adjacent c_8_0 c_8_1)
        (adjacent c_7_2 c_7_3)
        (adjacent c_9_13 c_8_13)
        (adjacent c_5_10 c_5_11)
        (adjacent c_0_6 c_0_7)
        (adjacent c_8_2 c_9_2)
        (adjacent c_9_5 c_10_5)
        (adjacent c_10_5 c_9_5)
        (adjacent c_9_9 c_10_9)
        (adjacent c_11_6 c_11_5)
        (adjacent c_0_7 c_1_7)
        (adjacent c_9_11 c_9_12)
        (adjacent c_8_5 c_9_5)
        (adjacent c_10_13 c_10_12)
        (adjacent c_2_12 c_2_11)
        (adjacent c_7_1 c_8_1)
        (adjacent c_6_6 c_6_7)
        (adjacent c_9_8 c_9_9)
        (adjacent c_6_10 c_7_10)
        (adjacent c_5_11 c_5_10)
        (adjacent c_13_5 c_12_5)
        (adjacent c_4_6 c_5_6)
        (adjacent c_11_12 c_12_12)
        (adjacent c_1_6 c_1_5)
        (adjacent c_4_12 c_4_11)
        (adjacent c_12_4 c_12_3)
        (adjacent c_4_10 c_4_9)
        (adjacent c_0_5 c_1_5)
        (adjacent c_4_11 c_5_11)
        (adjacent c_6_5 c_5_5)
        (adjacent c_8_6 c_8_7)
        (adjacent c_5_7 c_5_6)
        (adjacent c_4_2 c_4_1)
        (adjacent c_5_13 c_5_12)
        (adjacent c_12_3 c_12_2)
        (adjacent c_13_10 c_13_9)
        (adjacent c_6_6 c_5_6)
        (adjacent c_14_11 c_13_11)
        (adjacent c_6_1 c_6_2)
        (adjacent c_11_12 c_11_13)
        (adjacent c_4_6 c_4_5)
        (adjacent c_12_7 c_11_7)
        (adjacent c_3_7 c_2_7)
        (adjacent c_4_3 c_4_2)
        (adjacent c_10_12 c_10_13)
        (adjacent c_3_14 c_3_13)
        (adjacent c_6_3 c_6_2)
        (adjacent c_10_8 c_10_9)
        (adjacent c_0_8 c_0_9)
        (adjacent c_3_5 c_3_4)
        (adjacent c_7_6 c_8_6)
        (adjacent c_12_11 c_13_11)
        (adjacent c_5_12 c_5_13)
        (adjacent c_9_1 c_9_0)
        (adjacent c_7_1 c_6_1)
        (adjacent c_12_4 c_11_4)
        (adjacent c_14_10 c_14_9)
        (adjacent c_13_5 c_13_6)
        (adjacent c_11_2 c_11_1)
        (adjacent c_13_10 c_13_11)
        (adjacent c_1_10 c_0_10)
        (adjacent c_1_0 c_1_1)
        (adjacent c_3_5 c_2_5)
        (adjacent c_8_11 c_9_11)
        (adjacent c_3_4 c_3_3)
        (adjacent c_3_13 c_2_13)
        (adjacent c_9_4 c_8_4)
        (adjacent c_10_14 c_11_14)
        (adjacent c_13_7 c_12_7)
        (adjacent c_13_0 c_12_0)
        (adjacent c_4_9 c_4_10)
        (adjacent c_13_1 c_13_0)
        (adjacent c_4_7 c_5_7)
        (adjacent c_11_13 c_11_14)
        (adjacent c_8_14 c_8_13)
        (adjacent c_12_6 c_11_6)
        (adjacent c_6_10 c_6_9)
        (adjacent c_8_3 c_9_3)
        (adjacent c_3_6 c_4_6)
        (adjacent c_9_2 c_8_2)
        (adjacent c_12_10 c_13_10)
        (adjacent c_1_8 c_1_7)
        (adjacent c_13_4 c_14_4)
        (adjacent c_0_3 c_0_2)
        (adjacent c_7_6 c_7_5)
        (adjacent c_13_11 c_13_12)
        (adjacent c_5_6 c_5_7)
        (adjacent c_7_5 c_7_6)
        (adjacent c_4_4 c_3_4)
        (adjacent c_6_3 c_7_3)
        (adjacent c_12_12 c_12_11)
        (adjacent c_12_11 c_12_10)
        (adjacent c_8_10 c_9_10)
        (adjacent c_8_1 c_8_2)
        (adjacent c_12_13 c_13_13)
        (adjacent c_2_8 c_3_8)
        (adjacent c_2_6 c_1_6)
        (adjacent c_13_10 c_12_10)
        (adjacent c_4_0 c_3_0)
        (adjacent c_6_7 c_5_7)
        (adjacent c_13_14 c_14_14)
        (adjacent c_14_9 c_14_10)
        (adjacent c_10_0 c_9_0)
        (adjacent c_3_11 c_4_11)
        (adjacent c_11_4 c_10_4)
        (adjacent c_7_7 c_7_8)
        (adjacent c_3_3 c_4_3)
        (adjacent c_5_0 c_5_1)
        (adjacent c_2_5 c_1_5)
        (adjacent c_11_4 c_11_3)
        (adjacent c_6_6 c_7_6)
        (adjacent c_7_11 c_7_12)
        (adjacent c_10_3 c_11_3)
        (adjacent c_0_12 c_0_13)
        (adjacent c_7_5 c_7_4)
        (adjacent c_13_9 c_13_8)
        (adjacent c_12_8 c_12_7)
        (adjacent c_9_13 c_9_14)
        (adjacent c_12_0 c_11_0)
        (adjacent c_13_13 c_13_12)
        (adjacent c_3_14 c_4_14)
        (adjacent c_6_7 c_6_8)
        (adjacent c_8_0 c_9_0)
        (adjacent c_7_4 c_8_4)
        (adjacent c_10_3 c_9_3)
        (adjacent c_11_10 c_11_11)
        (adjacent c_6_1 c_7_1)
        (adjacent c_6_9 c_6_8)
        (adjacent c_3_9 c_2_9)
        (adjacent c_5_6 c_4_6)
        (adjacent c_8_14 c_9_14)
        (adjacent c_3_12 c_2_12)
        (adjacent c_0_11 c_1_11)
        (adjacent c_7_9 c_7_10)
        (adjacent c_1_1 c_1_2)
        (adjacent c_4_3 c_5_3)
        (adjacent c_8_3 c_8_4)
        (adjacent c_10_4 c_9_4)
        (adjacent c_1_14 c_0_14)
        (adjacent c_6_10 c_5_10)
        (adjacent c_0_9 c_0_8)
        (adjacent c_3_8 c_3_9)
        (adjacent c_6_11 c_6_10)
        (adjacent c_12_1 c_13_1)
        (adjacent c_9_10 c_9_11)
        (adjacent c_12_5 c_11_5)
        (adjacent c_13_1 c_12_1)
        (adjacent c_13_2 c_12_2)
        (adjacent c_13_3 c_12_3)
        (adjacent c_5_14 c_5_13)
        (adjacent c_6_13 c_5_13)
        (adjacent c_13_9 c_13_10)
        (adjacent c_0_1 c_0_0)
        (adjacent c_12_5 c_12_6)
        (adjacent c_3_12 c_4_12)
        (adjacent c_7_11 c_6_11)
        (adjacent c_13_12 c_13_11)
        (adjacent c_1_4 c_2_4)
        (adjacent c_9_10 c_9_9)
        (adjacent c_3_2 c_4_2)
        (adjacent c_3_6 c_3_7)
        (adjacent c_11_1 c_12_1)
        (adjacent c_7_3 c_7_4)
        (adjacent c_12_14 c_13_14)
        (adjacent c_7_14 c_8_14)
        (adjacent c_5_10 c_4_10)
        (adjacent c_8_7 c_8_8)
        (adjacent c_9_5 c_9_6)
        (adjacent c_14_13 c_13_13)
        (adjacent c_7_11 c_8_11)
        (adjacent c_5_4 c_6_4)
        (adjacent c_6_11 c_5_11)
        (adjacent c_11_8 c_12_8)
        (adjacent c_12_13 c_11_13)
        (adjacent c_14_3 c_13_3)
        (adjacent c_9_7 c_9_8)
        (adjacent c_3_4 c_2_4)
        (adjacent c_1_5 c_0_5)
        (adjacent c_6_4 c_6_3)
        (adjacent c_12_4 c_13_4)
        (adjacent c_10_5 c_10_6)
        (adjacent c_1_4 c_0_4)
        (adjacent c_11_0 c_12_0)
        (adjacent c_5_1 c_6_1)
        (adjacent c_11_9 c_11_10)
        (adjacent c_10_10 c_11_10)
        (adjacent c_0_4 c_1_4)
        (adjacent c_5_12 c_6_12)
        (adjacent c_3_4 c_4_4)
        (adjacent c_4_1 c_4_2)
        (adjacent c_8_13 c_7_13)
        (adjacent c_9_10 c_10_10)
        (adjacent c_7_4 c_6_4)
        (adjacent c_1_12 c_2_12)
        (adjacent c_5_3 c_5_2)
        (adjacent c_4_5 c_3_5)
        (adjacent c_12_8 c_12_9)
        (adjacent c_8_7 c_7_7)
        (adjacent c_0_5 c_0_6)
        (adjacent c_10_4 c_10_5)
        (adjacent c_3_2 c_3_1)
        (adjacent c_10_9 c_11_9)
        (adjacent c_12_2 c_12_3)
        (adjacent c_0_10 c_1_10)
        (adjacent c_10_13 c_9_13)
        (adjacent c_10_6 c_10_5)
        (adjacent c_9_3 c_10_3)
        (adjacent c_1_13 c_1_12)
        (adjacent c_13_6 c_13_5)
        (adjacent c_10_1 c_9_1)
        (adjacent c_10_7 c_10_8)
        (adjacent c_10_8 c_10_7)
        (adjacent c_0_9 c_0_10)
        (adjacent c_1_1 c_2_1)
        (adjacent c_7_10 c_7_9)
        (adjacent c_10_9 c_9_9)
        (adjacent c_1_14 c_2_14)
        (adjacent c_8_10 c_8_9)
        (adjacent c_2_5 c_3_5)
        (adjacent c_14_12 c_13_12)
        (adjacent c_7_4 c_7_5)
        (adjacent c_12_2 c_11_2)
        (adjacent c_11_8 c_10_8)
        (adjacent c_6_6 c_6_5)
        (adjacent c_11_10 c_10_10)
        (adjacent c_10_2 c_11_2)
        (adjacent c_13_4 c_13_3)
        (adjacent c_2_9 c_1_9)
        (adjacent c_5_0 c_4_0)
        (adjacent c_6_8 c_7_8)
        (adjacent c_10_13 c_10_14)
        (adjacent c_11_7 c_12_7)
        (adjacent c_8_14 c_7_14)
        (adjacent c_10_5 c_10_4)
        (adjacent c_5_1 c_4_1)
        (adjacent c_8_8 c_9_8)
        (adjacent c_0_4 c_0_3)
        (adjacent c_2_14 c_3_14)
        (adjacent c_11_1 c_10_1)
        (adjacent c_12_3 c_12_4)
        (adjacent c_1_3 c_2_3)
        (adjacent c_12_10 c_12_9)
        (adjacent c_1_5 c_1_4)
        (adjacent c_1_11 c_1_10)
        (adjacent c_2_10 c_2_11)
        (adjacent c_1_9 c_1_10)
        (adjacent c_5_3 c_6_3)
        (adjacent c_0_2 c_0_3)
        (adjacent c_12_1 c_12_0)
        (adjacent c_6_11 c_6_12)
        (adjacent c_14_9 c_13_9)
        (adjacent c_7_1 c_7_2)
        (adjacent c_13_8 c_13_9)
        (adjacent c_3_13 c_3_14)
        (adjacent c_3_6 c_3_5)
        (adjacent c_9_1 c_10_1)
        (adjacent c_10_12 c_9_12)
        (adjacent c_13_13 c_12_13)
        (adjacent c_5_3 c_5_4)
        (adjacent c_10_6 c_9_6)
        (adjacent c_7_13 c_7_12)
        (adjacent c_3_10 c_4_10)
        (adjacent c_9_14 c_8_14)
        (adjacent c_0_10 c_0_9)
        (adjacent c_4_8 c_4_7)
        (adjacent c_7_2 c_6_2)
        (adjacent c_14_1 c_13_1)
        (adjacent c_9_0 c_10_0)
        (adjacent c_7_1 c_7_0)
        (adjacent c_1_7 c_0_7)
        (adjacent c_4_13 c_3_13)
        (adjacent c_1_12 c_1_13)
        (adjacent c_13_12 c_13_13)
        (adjacent c_12_3 c_13_3)
        (adjacent c_1_5 c_1_6)
        (adjacent c_4_13 c_5_13)
        (adjacent c_5_5 c_6_5)
        (adjacent c_1_7 c_2_7)
        (adjacent c_4_3 c_3_3)
        (adjacent c_5_4 c_4_4)
        (adjacent c_14_5 c_14_6)
        (adjacent c_5_5 c_4_5)
        (adjacent c_8_9 c_8_10)
        (adjacent c_12_10 c_12_11)
        (adjacent c_3_7 c_3_6)
        (adjacent c_11_3 c_10_3)
        (adjacent c_12_7 c_12_6)
        (adjacent c_13_11 c_12_11)
        (adjacent c_12_13 c_12_14)
        (adjacent c_13_14 c_12_14)
        (adjacent c_14_10 c_14_11)
        (adjacent c_11_6 c_10_6)
        (adjacent c_10_2 c_10_1)
        (adjacent c_1_13 c_1_14)
        (adjacent c_5_3 c_4_3)
        (adjacent c_3_9 c_3_8)
        (adjacent c_3_5 c_4_5)
        (adjacent c_14_2 c_14_3)
        (adjacent c_1_2 c_0_2)
        (adjacent c_2_1 c_2_0)
        (adjacent c_3_6 c_2_6)
        (adjacent c_4_10 c_5_10)
        (adjacent c_10_14 c_10_13)
        (adjacent c_4_13 c_4_12)
        (adjacent c_14_13 c_14_12)
        (adjacent c_2_6 c_3_6)
        (adjacent c_12_0 c_12_1)
        (adjacent c_4_7 c_4_6)
        (adjacent c_4_13 c_4_14)
        (adjacent c_7_14 c_7_13)
        (adjacent c_0_0 c_1_0)
        (adjacent c_14_1 c_14_2)
        (adjacent c_13_1 c_14_1)
        (adjacent c_8_12 c_8_13)
        (adjacent c_3_10 c_3_9)
        (adjacent c_9_5 c_9_4)
        (adjacent c_6_8 c_6_7)
        (adjacent c_7_14 c_6_14)
        (adjacent c_2_7 c_2_8)
        (adjacent c_10_7 c_9_7)
        (adjacent c_2_10 c_2_9)
        (adjacent c_10_8 c_11_8)
        (adjacent c_12_4 c_12_5)
        (adjacent c_2_11 c_3_11)
        (adjacent c_8_9 c_8_8)
        (adjacent c_10_10 c_10_9)
        (adjacent c_8_13 c_8_14)
        (adjacent c_5_7 c_5_8)
        (adjacent c_9_4 c_10_4)
        (adjacent c_14_1 c_14_0)
        (adjacent c_10_12 c_10_11)
        (adjacent c_0_1 c_1_1)
        (adjacent c_0_2 c_1_2)
        (adjacent c_1_0 c_2_0)
        (adjacent c_2_7 c_2_6)
        (adjacent c_5_5 c_5_6)
        (adjacent c_6_3 c_6_4)
        (adjacent c_0_13 c_1_13)
        (adjacent c_5_10 c_5_9)
        (adjacent c_8_2 c_7_2)
        (adjacent c_8_3 c_8_2)
        (adjacent c_8_4 c_8_3)
        (adjacent c_1_4 c_1_3)
        (adjacent c_12_10 c_11_10)
        (adjacent c_5_2 c_6_2)
        (adjacent c_9_2 c_10_2)
        (adjacent c_4_2 c_4_3)
        (adjacent c_9_8 c_9_7)
        (adjacent c_10_9 c_10_10)
        (adjacent c_6_14 c_7_14)
        (adjacent c_14_10 c_13_10)
        (adjacent c_9_3 c_9_2)
        (adjacent c_8_4 c_8_5)
        (adjacent c_2_4 c_1_4)
        (adjacent c_12_7 c_13_7)
        (adjacent c_14_4 c_14_3)
        (adjacent c_10_0 c_11_0)
        (adjacent c_5_12 c_5_11)
        (adjacent c_5_5 c_5_4)
        (adjacent c_13_14 c_13_13)
        (adjacent c_10_11 c_10_10)
        (adjacent c_0_2 c_0_1)
        (adjacent c_8_9 c_9_9)
        (adjacent c_1_6 c_1_7)
        (adjacent c_12_8 c_11_8)
        (adjacent c_9_8 c_10_8)
        (adjacent c_0_13 c_0_14)
        (adjacent c_0_8 c_0_7)
        (adjacent c_3_5 c_3_6)
        (adjacent c_7_8 c_6_8)
        (adjacent c_14_12 c_14_13)
        (adjacent c_6_0 c_6_1)
        (adjacent c_9_9 c_9_10)
        (adjacent c_3_7 c_4_7)
        (adjacent c_7_3 c_7_2)
        (adjacent c_9_11 c_10_11)
        (adjacent c_4_6 c_4_7)
        (adjacent c_4_14 c_4_13)
        (adjacent c_10_12 c_11_12)
        (adjacent c_13_4 c_12_4)
        (adjacent c_14_14 c_13_14)
        (adjacent c_12_1 c_11_1)
        (adjacent c_2_0 c_3_0)
        (adjacent c_0_10 c_0_11)
        (adjacent c_1_10 c_1_11)
        (adjacent c_12_11 c_12_12)
        (adjacent c_2_10 c_1_10)
        (adjacent c_1_11 c_2_11)
        (adjacent c_4_0 c_4_1)
        (adjacent c_5_11 c_5_12)
        (adjacent c_14_6 c_13_6)
        (adjacent c_3_10 c_3_11)
        (adjacent c_7_12 c_7_11)
        (adjacent c_7_4 c_7_3)
        (adjacent c_9_4 c_9_3)
        (adjacent c_7_13 c_6_13)
        (adjacent c_10_4 c_11_4)
        (adjacent c_7_6 c_7_7)
        (adjacent c_11_11 c_11_12)
        (adjacent c_13_8 c_12_8)
        (adjacent c_8_11 c_7_11)
        (adjacent c_0_11 c_0_12)
        (adjacent c_3_10 c_2_10)
        (adjacent c_4_9 c_5_9)
        (adjacent c_9_10 c_8_10)
        (adjacent c_2_14 c_1_14)
        (adjacent c_13_10 c_14_10)
        (adjacent c_11_3 c_12_3)
        (adjacent c_13_3 c_13_4)
        (adjacent c_5_0 c_6_0)
        (adjacent c_1_7 c_1_6)
        (adjacent c_3_2 c_2_2)
        (adjacent c_9_6 c_10_6)
        (adjacent c_7_0 c_7_1)
        (adjacent c_4_7 c_4_8)
        (adjacent c_9_13 c_10_13)
        (adjacent c_11_1 c_11_2)
        (adjacent c_6_13 c_6_12)
        (adjacent c_2_13 c_2_12)
        (adjacent c_12_7 c_12_8)
        (adjacent c_3_13 c_4_13)
        (adjacent c_13_11 c_13_10)
        (adjacent c_2_11 c_2_10)
        (adjacent c_5_7 c_6_7)
        (adjacent c_12_14 c_12_13)
        (adjacent c_2_0 c_1_0)
        (adjacent c_0_3 c_1_3)
        (adjacent c_8_13 c_9_13)
        (adjacent c_2_9 c_2_10)
        (adjacent c_6_9 c_5_9)
        (adjacent c_5_8 c_5_7)
        (adjacent c_6_2 c_6_1)
        (adjacent c_11_3 c_11_2)
        (adjacent c_11_4 c_11_5)
        (adjacent c_2_6 c_2_7)
        (adjacent c_1_11 c_1_12)
        (adjacent c_12_5 c_13_5)
        (adjacent c_8_2 c_8_3)
        (adjacent c_6_10 c_6_11)
        (adjacent c_8_7 c_9_7)
        (adjacent c_14_7 c_14_8)
        (adjacent c_3_11 c_2_11)
        (adjacent c_11_11 c_10_11)
        (adjacent c_14_9 c_14_8)
        (adjacent c_3_9 c_4_9)
        (adjacent c_2_12 c_1_12)
        (adjacent c_4_12 c_5_12)
        (adjacent c_4_9 c_3_9)
        (adjacent c_7_9 c_6_9)
        (adjacent c_4_12 c_4_13)
        (adjacent c_10_3 c_10_4)
        (adjacent c_9_9 c_9_8)
        (adjacent c_6_8 c_6_9)
        (adjacent c_2_3 c_2_4)
        (adjacent c_1_9 c_1_8)
        (adjacent c_11_6 c_12_6)
        (adjacent c_2_3 c_1_3)
        (adjacent c_11_0 c_11_1)
        (adjacent c_10_6 c_10_7)
        (adjacent c_3_12 c_3_11)
        (adjacent c_5_1 c_5_0)
        (adjacent c_7_8 c_7_7)
        (adjacent c_4_2 c_5_2)
        (adjacent c_8_12 c_7_12)
        (adjacent c_3_4 c_3_5)
        (adjacent c_4_11 c_4_10)
        (adjacent c_11_6 c_11_7)
        (adjacent c_7_3 c_6_3)
        (adjacent c_2_4 c_2_5)
        (adjacent c_0_8 c_1_8)
        (adjacent c_5_2 c_5_3)
        (adjacent c_10_13 c_11_13)
        (adjacent c_4_11 c_4_12)
        (adjacent c_2_12 c_3_12)
        (adjacent c_7_10 c_6_10)
        (adjacent c_4_4 c_5_4)
        (adjacent c_11_5 c_10_5)
        (adjacent c_9_7 c_8_7)
        (adjacent c_4_9 c_4_8)
        (adjacent c_7_5 c_8_5)
        (adjacent c_6_4 c_7_4)
        (adjacent c_11_5 c_11_6)
        (adjacent c_1_11 c_0_11)
        (adjacent c_3_3 c_3_2)
        (adjacent c_5_14 c_6_14)
        (adjacent c_6_13 c_6_14)
        (adjacent c_7_13 c_8_13)
        (adjacent c_1_2 c_2_2)
        (adjacent c_13_0 c_13_1)
        (adjacent c_8_10 c_7_10)
        (adjacent c_8_9 c_7_9)
        (adjacent c_11_4 c_12_4)
        (adjacent c_10_0 c_10_1)
        (adjacent c_11_1 c_11_0)
        (adjacent c_5_11 c_4_11)
        (adjacent c_2_11 c_1_11)
        (adjacent c_2_12 c_2_13)
        (adjacent c_0_0 c_0_1)
        (adjacent c_3_1 c_3_2)
        (adjacent c_13_6 c_14_6)
        (adjacent c_8_11 c_8_10)
        (adjacent c_2_5 c_2_4)
        (adjacent c_7_10 c_8_10)
        (adjacent c_4_5 c_4_6)
        (adjacent c_2_3 c_3_3)
        (adjacent c_2_13 c_2_14)
        (adjacent c_5_8 c_6_8)
        (adjacent c_4_7 c_3_7)
        (adjacent c_2_9 c_3_9)
        (adjacent c_10_2 c_9_2)
        (adjacent c_6_0 c_5_0)
        (adjacent c_4_10 c_3_10)
        (adjacent c_6_4 c_5_4)
        (adjacent c_1_12 c_1_11)
        (adjacent c_2_2 c_3_2)
        (adjacent c_1_6 c_0_6)
        (adjacent c_10_14 c_9_14)
        (adjacent c_5_4 c_5_5)
        (adjacent c_14_11 c_14_10)
        (adjacent c_0_6 c_1_6)
        (adjacent c_2_8 c_2_7)
        (adjacent c_6_5 c_6_6)
        (adjacent c_6_0 c_7_0)
        (adjacent c_7_0 c_6_0)
        (adjacent c_12_6 c_13_6)
        (adjacent c_10_2 c_10_3)
        (adjacent c_8_4 c_9_4)
        (adjacent c_14_12 c_14_11)
        (adjacent c_3_7 c_3_8)
        (adjacent c_9_14 c_10_14)
        (adjacent c_7_9 c_7_8)
        (adjacent c_12_12 c_11_12)
        (adjacent c_7_12 c_8_12)
        (adjacent c_5_6 c_5_5)
        (adjacent c_10_1 c_10_0)
        (adjacent c_12_2 c_12_1)
        (adjacent c_1_13 c_0_13)
        (adjacent c_1_10 c_1_9)
        (adjacent c_5_8 c_4_8)
        (adjacent c_8_8 c_8_9)
        (adjacent c_1_1 c_0_1)
        (adjacent c_5_13 c_4_13)
        (adjacent c_7_12 c_7_13)
        (adjacent c_10_3 c_10_2)
        (adjacent c_11_2 c_10_2)
        (adjacent c_9_14 c_9_13)
        (adjacent c_12_9 c_12_10)
        (adjacent c_4_14 c_3_14)
        (adjacent c_9_7 c_9_6)
        (adjacent c_13_5 c_14_5)
        (adjacent c_3_13 c_3_12)
        (adjacent c_1_4 c_1_5)
        (adjacent c_1_7 c_1_8)
        (adjacent c_14_7 c_13_7)
        (adjacent c_0_7 c_0_8)
        (adjacent c_9_11 c_9_10)
        (adjacent c_12_8 c_13_8)
        (adjacent c_5_2 c_4_2)
        (adjacent c_14_8 c_13_8)
        (adjacent c_6_4 c_6_5)
        (adjacent c_8_2 c_8_1)
        (adjacent c_4_1 c_5_1)
        (adjacent c_1_14 c_1_13)
        (adjacent c_8_4 c_7_4)
        (adjacent c_5_11 c_6_11)
        (adjacent c_10_1 c_10_2)
        (adjacent c_14_11 c_14_12)
        (adjacent c_0_6 c_0_5)
        (adjacent c_8_1 c_7_1)
        (adjacent c_1_3 c_1_4)
        (adjacent c_0_13 c_0_12)
        (adjacent c_10_7 c_11_7)
        (adjacent c_13_13 c_13_14)
        (adjacent c_6_14 c_5_14)
        (adjacent c_6_14 c_6_13)
        (adjacent c_1_0 c_0_0)
        (adjacent c_4_4 c_4_5)
        (adjacent c_8_0 c_7_0)
        (adjacent c_13_2 c_13_3)
        (adjacent c_13_6 c_13_7)
        (adjacent c_1_13 c_2_13)
        (adjacent c_3_8 c_2_8)
        (adjacent c_6_2 c_5_2)
        (adjacent c_14_2 c_14_1)
        (adjacent c_3_8 c_4_8)
        (adjacent c_14_14 c_14_13)
        (adjacent c_9_7 c_10_7)
        (adjacent c_7_7 c_8_7)
        (adjacent c_1_8 c_1_9)
        (adjacent c_9_2 c_9_3)
        (adjacent c_8_6 c_9_6)
        (adjacent c_6_9 c_7_9)
        (adjacent c_9_12 c_10_12)
        (adjacent c_3_1 c_4_1)
        (adjacent c_13_6 c_12_6)
        (adjacent c_5_1 c_5_2)
        (adjacent c_1_5 c_2_5)
        (adjacent c_0_14 c_0_13)
        (adjacent c_11_13 c_11_12)
        (adjacent c_13_7 c_14_7)
        (adjacent c_3_9 c_3_10)
        (adjacent c_6_5 c_7_5)
        (adjacent c_8_1 c_8_0)
        (adjacent c_4_14 c_5_14)
        (adjacent c_4_4 c_4_3)
        (adjacent c_6_12 c_6_11)
        (adjacent c_2_7 c_1_7)
        (adjacent c_1_9 c_2_9)
        (adjacent c_8_1 c_9_1)
        (adjacent c_1_6 c_2_6)
        (adjacent c_5_4 c_5_3)
        (adjacent c_12_12 c_13_12)
        (adjacent c_13_1 c_13_2)
        (adjacent c_14_4 c_14_5)
        (adjacent c_12_1 c_12_2)
        (adjacent c_13_2 c_13_1)
        (adjacent c_9_6 c_9_5)
        (adjacent c_8_12 c_9_12)
        (adjacent c_13_7 c_13_6)
        (adjacent c_3_0 c_3_1)
        (adjacent c_1_3 c_1_2)
        (adjacent c_9_12 c_9_13)
        (adjacent c_11_8 c_11_7)
        (adjacent c_3_14 c_2_14)
        (adjacent c_10_7 c_10_6)
        (adjacent c_0_1 c_0_2)
        (adjacent c_10_4 c_10_3)
        (adjacent c_10_5 c_11_5)
        (adjacent c_0_7 c_0_6)
        (adjacent c_2_14 c_2_13)
        (adjacent c_9_0 c_8_0)
        (adjacent c_13_9 c_12_9)
        (adjacent c_13_12 c_14_12)
        (adjacent c_2_6 c_2_5)
        (adjacent c_6_7 c_7_7)
        (adjacent c_1_8 c_2_8)
        (adjacent c_5_8 c_5_9)
        (adjacent c_12_6 c_12_5)
        (adjacent c_7_5 c_6_5)
        (adjacent c_2_11 c_2_12)
        (adjacent c_14_4 c_13_4)
        (adjacent c_5_10 c_6_10)
        (adjacent c_11_2 c_12_2)
        (adjacent c_2_8 c_2_9)
        (adjacent c_8_3 c_7_3)
        (adjacent c_10_10 c_9_10)
        (adjacent c_12_2 c_13_2)
        (adjacent c_13_8 c_14_8)
        (adjacent c_9_12 c_8_12)
        (adjacent c_3_12 c_3_13)
        (adjacent c_8_8 c_8_7)
        (adjacent c_3_2 c_3_3)
        (adjacent c_5_2 c_5_1)
        (adjacent c_7_2 c_8_2)
        (adjacent c_7_13 c_7_14)
        (adjacent c_9_1 c_9_2)
        (adjacent c_11_8 c_11_9)
        (adjacent c_11_5 c_11_4)
        (adjacent c_12_14 c_11_14)
        (adjacent c_2_10 c_3_10)
        (adjacent c_14_3 c_14_4)
        (adjacent c_13_11 c_14_11)
        (adjacent c_4_2 c_3_2)
        (adjacent c_8_13 c_8_12)
        (adjacent c_14_2 c_13_2)
        (adjacent c_10_1 c_11_1)
        (adjacent c_14_5 c_14_4)
        (adjacent c_9_3 c_9_4)
        (adjacent c_3_11 c_3_12)
        (adjacent c_12_13 c_12_12)
        (adjacent c_11_7 c_10_7)
        (adjacent c_14_3 c_14_2)
        (adjacent c_5_12 c_4_12)
        (adjacent c_2_13 c_3_13)
        (adjacent c_2_4 c_3_4)
        (adjacent c_11_14 c_12_14)
        (adjacent c_2_7 c_3_7)
        (adjacent c_11_13 c_12_13)
        (adjacent c_8_8 c_7_8)
        (adjacent c_5_9 c_5_8)
        (adjacent c_0_14 c_1_14)
        (adjacent c_1_3 c_0_3)
        (adjacent c_0_12 c_1_12)
        (adjacent c_7_12 c_6_12)
        (adjacent c_9_8 c_8_8)
        (adjacent c_7_6 c_6_6)
        (adjacent c_8_7 c_8_6)
        (adjacent c_8_12 c_8_11)
        (adjacent c_12_0 c_13_0)
        (adjacent c_8_5 c_7_5)
        (adjacent c_11_5 c_12_5)
        (adjacent c_2_4 c_2_3)
        (adjacent c_2_1 c_3_1)
        (adjacent c_4_8 c_3_8)
        (adjacent c_6_1 c_5_1)
        (adjacent c_4_1 c_3_1)
        (adjacent c_7_8 c_8_8)
        (adjacent c_10_8 c_9_8)
        (adjacent c_11_2 c_11_3)
        (adjacent c_14_7 c_14_6)
        (adjacent c_5_9 c_4_9)
        (adjacent c_4_10 c_4_11)
        (adjacent c_0_9 c_1_9)
        (adjacent c_1_2 c_1_1)
        (adjacent c_11_3 c_11_4)
        (adjacent c_13_3 c_13_2)
        (adjacent c_7_2 c_7_1)
        (adjacent c_13_5 c_13_4)
        (adjacent c_2_2 c_2_1)
        (adjacent c_7_11 c_7_10)
        (adjacent c_8_11 c_8_12)
        (adjacent c_14_6 c_14_5)
        (adjacent c_4_0 c_5_0)
        (adjacent c_11_7 c_11_8)
        (adjacent c_11_9 c_11_8)
        (adjacent c_0_3 c_0_4)
        (adjacent c_2_5 c_2_6)
        (adjacent c_6_12 c_7_12)
        (adjacent c_10_10 c_10_11)
        (adjacent c_2_1 c_1_1)
        (adjacent c_11_12 c_11_11)
        (adjacent c_12_9 c_13_9)
        (adjacent c_7_10 c_7_11)
        (adjacent c_9_11 c_8_11)
        (adjacent c_12_11 c_11_11)
        (adjacent c_0_11 c_0_10)
        (adjacent c_7_3 c_8_3)
        (adjacent c_12_9 c_11_9)
        (at c_0_0)
        (unblocked c_11_0)
        (unblocked c_12_0)
        (unblocked c_7_6)
        (unblocked c_6_4)
        (unblocked c_5_4)
        (unblocked c_11_12)
        (unblocked c_10_2)
        (unblocked c_7_0)
        (unblocked c_6_0)
        (unblocked c_11_13)
        (unblocked c_12_2)
        (unblocked c_0_11)
        (unblocked c_3_2)
        (unblocked c_1_12)
        (unblocked c_10_14)
        (unblocked c_4_5)
        (unblocked c_5_13)
        (unblocked c_6_6)
        (unblocked c_11_4)
        (unblocked c_8_0)
        (unblocked c_3_4)
        (unblocked c_5_0)
        (unblocked c_14_3)
        (unblocked c_10_4)
        (unblocked c_4_10)
        (unblocked c_11_5)
        (unblocked c_12_8)
        (unblocked c_0_9)
        (unblocked c_12_1)
        (unblocked c_9_9)
        (unblocked c_14_4)
        (unblocked c_5_8)
        (unblocked c_1_10)
        (unblocked c_9_14)
        (unblocked c_13_5)
        (unblocked c_9_2)
        (unblocked c_7_10)
        (unblocked c_7_9)
        (unblocked c_1_9)
        (unblocked c_10_3)
        (unblocked c_2_10)
        (unblocked c_10_12)
        (unblocked c_7_12)
        (unblocked c_4_8)
        (unblocked c_14_5)
        (unblocked c_10_11)
        (unblocked c_0_8)
        (unblocked c_9_12)
        (unblocked c_12_10)
        (unblocked c_6_11)
        (unblocked c_10_0)
        (unblocked c_1_7)
        (unblocked c_0_5)
        (unblocked c_13_0)
        (unblocked c_3_11)
        (unblocked c_1_4)
        (unblocked c_10_8)
        (unblocked c_11_10)
        (unblocked c_4_6)
        (unblocked c_9_8)
        (unblocked c_7_13)
        (unblocked c_7_5)
        (unblocked c_0_3)
        (unblocked c_2_1)
        (unblocked c_13_3)
        (unblocked c_6_3)
        (unblocked c_10_9)
        (unblocked c_9_4)
        (unblocked c_8_5)
        (unblocked c_2_9)
        (unblocked c_2_13)
        (unblocked c_11_3)
        (unblocked c_11_1)
        (unblocked c_13_2)
        (unblocked c_4_0)
        (unblocked c_5_3)
        (unblocked c_13_1)
        (unblocked c_0_1)
        (unblocked c_2_8)
        (unblocked c_12_9)
        (unblocked c_12_4)
        (unblocked c_13_7)
        (unblocked c_3_3)
        (unblocked c_3_1)
        (unblocked c_12_13)
        (unblocked c_13_6)
        (unblocked c_0_0)
        (unblocked c_8_4)
        (unblocked c_9_0)
        (unblocked c_9_7)
        (unblocked c_13_9)
        (unblocked c_3_8)
        (unblocked c_4_11)
        (unblocked c_13_12)
        (unblocked c_5_11)
        (unblocked c_14_8)
        (unblocked c_1_2)
        (unblocked c_7_8)
        (unblocked c_9_13)
        (unblocked c_6_12)
        (unblocked c_1_5)
        (unblocked c_14_7)
        (unblocked c_13_8)
        (unblocked c_9_3)
        (unblocked c_11_6)
        (unblocked c_8_11)
        (unblocked c_0_10)
        (unblocked c_7_4)
        (unblocked c_11_2)
        (unblocked c_1_13)
        (unblocked c_2_11)
        (unblocked c_1_1)
        (unblocked c_3_13)
        (unblocked c_14_0)
        (unblocked c_11_7)
        (unblocked c_1_3)
        (unblocked c_8_8)
        (unblocked c_0_14)
        (unblocked c_2_3)
        (unblocked c_0_7)
        (unblocked c_3_7)
        (unblocked c_4_7)
        (unblocked c_8_10)
        (unblocked c_4_13)
        (unblocked c_8_7)
        (unblocked c_6_9)
        (unblocked c_6_14)
        (unblocked c_6_8)
        (unblocked c_11_9)
        (unblocked c_12_14)
        (unblocked c_0_6)
        (unblocked c_14_13)
        (unblocked c_14_11)
        (unblocked c_1_8)
        (unblocked c_8_9)
        (unblocked c_11_11)
        (unblocked c_8_14)
        (unblocked c_3_6)
        (unblocked c_13_13)
        (unblocked c_3_14)
        (unblocked c_12_7)
        (unblocked c_11_14)
        (unblocked c_10_1)
        (unblocked c_11_8)
        (unblocked c_14_6)
        (unblocked c_9_10)
        (unblocked c_5_2)
        (unblocked c_6_10)
        (unblocked c_5_5)
        (unblocked c_4_2)
        (unblocked c_7_11)
        (unblocked c_3_10)
        (unblocked c_5_14)
        (unblocked c_6_13)
        (unblocked c_13_11)
        (unblocked c_5_10)
        (unblocked c_10_7)
        (unblocked c_0_2)
        (unblocked c_14_2)
        (unblocked c_10_6)
        (unblocked c_1_11)
        (unblocked c_9_6)
        (unblocked c_8_13)
        (unblocked c_4_9)
        (unblocked c_8_6)
        (unblocked c_0_4)
        (unblocked c_9_1)
        (unblocked c_7_1)
        (unblocked c_14_9)
        (unblocked c_8_1)
        (unblocked c_6_1)
        (unblocked c_12_5)
        (unblocked c_3_5)
        (unblocked c_13_4)
        (unblocked c_4_12)
        (unblocked c_9_11)
        (unblocked c_12_3)
        (unblocked c_10_13)
        (unblocked c_14_10)
        (unblocked c_3_12)
        (unblocked c_13_14)
        (unblocked c_2_6)
        (unblocked c_14_1)
        (unblocked c_0_13)
        (unblocked c_2_12)
        (unblocked c_5_1)
        (unblocked c_5_12)
        (unblocked c_6_7)
        (unblocked c_7_2)
        (unblocked c_5_7)
        (unblocked c_1_0)
        (unblocked c_3_0)
        (unblocked c_7_3)
        (unblocked c_8_3)
        (unblocked c_4_4)
        (unblocked c_2_14)
        (unblocked c_12_11)
        (unblocked c_2_0)
        (unblocked c_2_5)
        (unblocked c_8_12)
        (unblocked c_6_5)
        (unblocked c_5_9)
        (unblocked c_4_14)
        (unblocked c_14_12)
        (unblocked c_5_6)
        (unblocked c_1_14)
        (unblocked c_2_4)
        (unblocked c_6_2)
        (unblocked c_10_10)
        (unblocked c_2_2)
        (unblocked c_12_6)
        (unblocked c_3_9)
        (reward c_3_7)
        (reward c_5_8)
        (reward c_5_14)
        (reward c_3_13)
        (reward c_10_1)
        (reward c_10_7)
        (reward c_0_2)
        (reward c_4_10)
        (reward c_0_1)
        (reward c_14_7)
        (reward c_11_14)
        (reward c_5_1)
        (reward c_8_5)
    )

    (:goal
        (and (picked c_3_7) (picked c_5_14) (picked c_0_1) (picked c_10_7) (picked c_5_1) (picked c_0_2) (picked c_14_7) (picked c_4_10) (picked c_10_1) (picked c_3_13) (picked c_8_5) (picked c_11_14) (picked c_5_8))
    )

    
    
    
)

