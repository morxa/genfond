(define (problem blocks-018-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b13) (clear b16) (clear b3) (clear b8) (handempty) (on b0 b17) (on b1 b9) (on b10 b4) (on b12 b2) (on b14 b1) (on b15 b6) (on b17 b10) (on b2 b0) (on b3 b7) (on b4 b5) (on b5 b11) (on b6 b12) (on b8 b14) (on b9 b15) (ontable b11) (ontable b13) (ontable b16) (ontable b7))
    (:goal (clear b15))
)