(define (problem blocks-018-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b11) (clear b13) (clear b14) (handempty) (on b0 b7) (on b11 b9) (on b13 b4) (on b14 b12) (on b16 b6) (on b17 b3) (on b2 b0) (on b3 b16) (on b4 b15) (on b5 b10) (on b6 b5) (on b7 b17) (on b8 b2) (on b9 b8) (ontable b1) (ontable b10) (ontable b12) (ontable b15))
    (:goal (on b12 b17))
)