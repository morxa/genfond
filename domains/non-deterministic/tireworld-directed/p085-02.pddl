(define (problem tireworld-085-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l9 - location)
    (:init (road l0 l14) (road l0 l25) (road l1 l29) (road l1 l84) (road l10 l37) (road l10 l77) (road l11 l39) (road l11 l7) (road l12 l52) (road l13 l39) (road l13 l50) (road l14 l0) (road l14 l28) (road l15 l30) (road l15 l61) (road l16 l37) (road l16 l70) (road l17 l3) (road l17 l56) (road l18 l23) (road l18 l58) (road l19 l81) (road l19 l9) (road l2 l34) (road l2 l35) (road l20 l31) (road l20 l67) (road l21 l26) (road l21 l74) (road l22 l44) (road l22 l64) (road l23 l18) (road l23 l33) (road l24 l62) (road l24 l79) (road l25 l0) (road l25 l76) (road l26 l21) (road l26 l61) (road l27 l40) (road l27 l50) (road l28 l14) (road l28 l67) (road l29 l1) (road l29 l68) (road l3 l17) (road l3 l83) (road l30 l15) (road l30 l45) (road l31 l20) (road l31 l42) (road l32 l48) (road l32 l68) (road l33 l23) (road l33 l55) (road l34 l2) (road l34 l77) (road l35 l2) (road l35 l84) (road l36 l71) (road l36 l8) (road l37 l10) (road l37 l16) (road l38 l54) (road l38 l68) (road l39 l11) (road l39 l13) (road l4 l57) (road l4 l9) (road l40 l27) (road l40 l82) (road l41 l5) (road l41 l54) (road l42 l31) (road l42 l80) (road l43 l53) (road l43 l65) (road l44 l22) (road l44 l69) (road l45 l30) (road l45 l46) (road l46 l45) (road l47 l5) (road l47 l72) (road l48 l32) (road l48 l73) (road l49 l74) (road l49 l75) (road l5 l41) (road l5 l47) (road l50 l13) (road l50 l27) (road l51 l6) (road l52 l12) (road l52 l72) (road l53 l43) (road l53 l83) (road l54 l38) (road l54 l41) (road l55 l33) (road l55 l60) (road l56 l17) (road l56 l7) (road l57 l4) (road l57 l79) (road l58 l18) (road l58 l71) (road l59 l65) (road l59 l78) (road l6 l51) (road l6 l66) (road l60 l55) (road l60 l76) (road l61 l15) (road l62 l24) (road l62 l73) (road l63 l64) (road l63 l8) (road l63 l82) (road l64 l22) (road l64 l63) (road l65 l43) (road l65 l59) (road l66 l6) (road l66 l80) (road l67 l20) (road l67 l28) (road l68 l29) (road l68 l32) (road l68 l38) (road l69 l12) (road l69 l44) (road l7 l11) (road l7 l56) (road l70 l16) (road l70 l78) (road l71 l36) (road l71 l58) (road l72 l47) (road l72 l52) (road l73 l48) (road l73 l62) (road l74 l21) (road l74 l49) (road l75 l49) (road l75 l81) (road l76 l25) (road l76 l60) (road l77 l10) (road l77 l34) (road l78 l59) (road l78 l70) (road l79 l24) (road l79 l57) (road l8 l36) (road l8 l63) (road l80 l42) (road l80 l66) (road l81 l19) (road l81 l75) (road l82 l40) (road l82 l63) (road l83 l3) (road l83 l53) (road l84 l1) (road l84 l35) (road l9 l19) (road l9 l4) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l38) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l9) (vehicle-at l51))
    (:goal (vehicle-at l46))
)