(define (problem blocks-018-5)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b4) (clear b8) (handempty) (on b1 b6) (on b10 b3) (on b11 b14) (on b12 b13) (on b13 b2) (on b14 b9) (on b15 b11) (on b16 b17) (on b17 b5) (on b2 b7) (on b3 b0) (on b4 b15) (on b5 b1) (on b6 b12) (on b7 b10) (on b9 b16) (ontable b0) (ontable b8))
    (:goal (and (clear b12) (on b12 b11) (on b11 b17) (on b17 b7) (on b7 b8) (on b8 b4) (on b4 b6) (on b6 b16) (on b16 b1) (ontable b1)))
)