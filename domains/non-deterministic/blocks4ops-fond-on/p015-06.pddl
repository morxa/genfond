(define (problem blocks-015-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b1) (clear b11) (clear b7) (handempty) (on b0 b12) (on b1 b5) (on b10 b6) (on b14 b4) (on b2 b3) (on b3 b9) (on b4 b2) (on b5 b13) (on b6 b14) (on b7 b8) (on b8 b10) (ontable b11) (ontable b12) (ontable b13) (ontable b9))
    (:goal (on b9 b6))
)