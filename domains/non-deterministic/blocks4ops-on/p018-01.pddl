(define (problem blocks-018-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b14) (clear b17) (clear b3) (handempty) (on b0 b12) (on b10 b15) (on b12 b16) (on b13 b2) (on b14 b13) (on b15 b11) (on b16 b8) (on b17 b10) (on b2 b9) (on b3 b5) (on b4 b7) (on b5 b4) (on b6 b1) (on b7 b0) (on b8 b6) (ontable b1) (ontable b11) (ontable b9))
    (:goal (on b16 b11))
)