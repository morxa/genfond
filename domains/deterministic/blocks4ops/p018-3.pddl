(define (problem blocks-018-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b14) (clear b15) (clear b16) (clear b4) (handempty) (on b0 b6) (on b1 b8) (on b10 b11) (on b11 b2) (on b13 b9) (on b16 b7) (on b17 b1) (on b2 b13) (on b3 b5) (on b4 b12) (on b7 b17) (on b8 b10) (on b9 b3) (ontable b12) (ontable b14) (ontable b15) (ontable b5) (ontable b6))
    (:goal (and (on b4 b11) (on b11 b15) (on b15 b10) (on b10 b8) (on b8 b12) (on b12 b0) (on b0 b3) (on b3 b17) (on b17 b2) (on b2 b5) (on b5 b13) (on b13 b14)))
)