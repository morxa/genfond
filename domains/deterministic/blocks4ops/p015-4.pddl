(define (problem blocks-015-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b12) (clear b14) (handempty) (on b0 b11) (on b1 b8) (on b10 b2) (on b11 b10) (on b12 b6) (on b13 b0) (on b2 b9) (on b4 b7) (on b5 b4) (on b6 b5) (on b7 b13) (on b9 b3) (ontable b14) (ontable b3) (ontable b8))
    (:goal (and (on b11 b1) (on b1 b6) (on b6 b5) (on b5 b7) (on b7 b0) (on b0 b9) (on b9 b4)))
)