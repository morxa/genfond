(define (problem blocks-018-10)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b10) (clear b16) (clear b3) (clear b6) (clear b7) (handempty) (on b0 b15) (on b1 b4) (on b11 b17) (on b12 b9) (on b13 b2) (on b14 b1) (on b15 b5) (on b16 b8) (on b17 b14) (on b2 b11) (on b6 b13) (on b7 b0) (on b8 b12) (ontable b10) (ontable b3) (ontable b4) (ontable b5) (ontable b9))
    (:goal (on b3 b11))
)